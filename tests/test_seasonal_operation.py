"""Seasonal input contracts and independent horizon-feasibility regressions."""
import copy
import csv
import hashlib
import io
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.optimize import linprog

from cascade_test_utils import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData, IDS, SEASONAL_FILES
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade, ReservoirPhysics
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.terminal_storage import TerminalReleaseBounds
from harl.envs.cascade_reservoir.evaluation import (
    SeasonalMetrics, evaluate_policy, reference_summary, validate_operating_feasibility)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/cascade_reservoir_new'
DT = 86400.


class SeasonalInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.physical = load_configuration()
        cls.data = CascadeData(DATA, cls.physical, cls.cfg['experiment'])
        cls.seasonal = cls.data.load_seasonal_operation()

    def temporary_data(self, root):
        for p in DATA.glob('*.csv'):
            shutil.copy2(p, root/p.name)
        return CascadeData(root, self.physical, self.cfg['experiment'])

    def index(self, day):
        return next(i for i,d in enumerate(self.data.dates) if d.isoformat() == day)

    def test_explicit_loading_and_provenance_preserve_existing_inputs(self):
        d = CascadeData(DATA, self.physical, self.cfg['experiment'])
        self.assertIsNone(d.seasonal_operation)
        before = [a.copy() for a in (d.vmin,d.vmax,d.flow_mean,d.flow_std)]
        s = d.load_seasonal_operation()
        self.assertIs(d.load_seasonal_operation(),s)
        for a,b in zip(before,(d.vmin,d.vmax,d.flow_mean,d.flow_std)):
            np.testing.assert_array_equal(a,b)
        self.assertEqual(s.storage_lower.shape,(21915,5))
        for name in SEASONAL_FILES:
            self.assertEqual(d.hashes[name],hashlib.sha256((DATA/name).read_bytes()).hexdigest())
        self.assertEqual(s.key_parameters['WDD']['value_status'],
                         'scenario_reference_not_official_operating_rule')
        self.assertEqual(len(s.period_parameters),20)
        self.assertEqual(sum(map(len,s.reference_nodes.values())),61)
        with self.assertRaises(ValueError):
            CascadeData(DATA,dict(self.physical,load_seasonal_operation='true'))

    def test_end_of_day_flood_ceiling_and_refill_deadline(self):
        d,s=self.data,self.seasonal
        checks=[('WDD','2023-05-31','2023-07-31','2023-10-31',952.,975.),
                ('BHT','2023-05-31','2023-07-31','2023-08-31',785.,825.),
                ('XLD','2023-05-31','2023-09-10','2023-09-30',560.,600.),
                ('XJB','2023-05-31','2023-09-10','2023-09-30',370.,380.),
                ('THR','2023-06-10','2023-09-10','2023-10-31',145.,175.)]
        for id,draw,end,refill,flood,target in checks:
            j=IDS.index(id);a,b,c=map(self.index,(draw,end,refill));p=d.anchors[id]
            v=float(np.interp(flood,p[:,0],p[:,1]))
            np.testing.assert_array_equal(s.storage_upper[a:b+1,j],np.full(b-a+1,v))
            self.assertEqual(s.storage_upper[a-1,j],d.vmax[j])
            self.assertEqual(s.storage_upper[b+1,j],d.vmax[j])
            self.assertFalse(s.reference['reference_flood_limit_active'][a,j])
            self.assertTrue(s.flood_limit_active[a,j])
            self.assertEqual(s.storage_lower[c,j],np.interp(target,p[:,0],p[:,1]))
            self.assertEqual(s.storage_lower[c-1,j],d.vmin[j])
            self.assertEqual(s.storage_lower[c+1,j],d.vmin[j])

    def test_soft_reference_is_not_a_hard_daily_bound(self):
        d,s=self.data,self.seasonal
        self.assertEqual(int((s.reference['reference_target_feasible_zero_lag']==0).sum()),31)
        # WDD's preferred band at the deadline extends above the hard ceiling.
        t=self.index('2023-05-31')
        self.assertEqual(s.reference['soft_target_upper_level_m'][t,0],953.)
        self.assertEqual(d.levels(s.storage_upper[t])[0],952.)
        # June 1 is constrained by the flood ceiling, not by the soft lower band.
        t=self.index('2023-06-01')
        self.assertEqual(s.storage_lower[t,0],d.vmin[0])
        self.assertEqual(s.reference_subphases[self.index('2023-08-01'),1],'refill')

    def test_leap_year_reference_and_returned_bounds_are_independent(self):
        d,s=self.data,self.seasonal
        t=self.index('2024-02-29')
        np.testing.assert_array_equal(s.reference['target_end_of_day_storage_m3'][t-1],
                                      s.reference['target_start_of_day_storage_m3'][t])
        for year,days in ((2023,365),(2024,366)):
            a,b=d.indices(year,year)
            low,high=s.storage_bounds(a,b)
            self.assertEqual(low.shape,(days,5))
            low[:]=0;high[:]=0
            self.assertTrue((s.storage_lower[a:b]>0).all())
        self.assertFalse(s.storage_lower.flags.writeable)
        self.assertFalse(s.reference['target_end_of_day_storage_m3'].flags.writeable)
        for args in ((-1,2),(2,2),(0,len(d.dates)+1),(True,2)):
            with self.assertRaises(ValueError):s.storage_bounds(*args)

    def test_small_tables_are_loaded_by_keys_not_row_position(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.temporary_data(root)
            for name in SEASONAL_FILES[:3]:
                with (root/name).open(encoding='utf-8-sig',newline='') as f:
                    rows=list(csv.DictReader(f))
                with (root/name).open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(reversed(rows))
            d=CascadeData(root,dict(self.physical,load_seasonal_operation=True),self.cfg['experiment'])
            np.testing.assert_array_equal(d.seasonal_operation.storage_upper,self.seasonal.storage_upper)
            np.testing.assert_array_equal(d.seasonal_operation.storage_lower,self.seasonal.storage_lower)
            # The daily source itself is alphabetically ordered, not hydraulic order.
            np.testing.assert_array_equal(d.seasonal_operation.reference['target_end_of_day_storage_m3'],
                                          self.seasonal.reference['target_end_of_day_storage_m3'])

    def test_missing_or_bad_key_data_fail_only_when_requested(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);d=self.temporary_data(root);path=root/SEASONAL_FILES[0]
            path.unlink()
            with self.assertRaisesRegex(ValueError,'requires.*five_key'):
                d.load_seasonal_operation()
            self.assertIsNone(d.seasonal_operation)
            self.assertFalse(any(name in d.hashes for name in SEASONAL_FILES))
            with (DATA/SEASONAL_FILES[0]).open(encoding='utf-8-sig',newline='') as f:
                original=list(csv.DictReader(f))
            changes=[('reservoir_id','BHT'),('flood_control_limit_level_m','nan'),
                     ('flood_control_limit_level_m','976'),('drawdown_deadline_mmdd','02-29'),
                     ('drawdown_deadline_mmdd','09-01'),('date_interpretation','start_of_day'),
                     ('impoundment_target_level_m',''),('value_status','')]
            for key,value in changes:
                rows=copy.deepcopy(original);rows[0][key]=value
                with path.open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    d.load_seasonal_operation()
                self.assertIsNone(d.seasonal_operation)

    def test_duplicate_missing_and_stale_daily_reference_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.temporary_data(root);path=root/SEASONAL_FILES[3]
            text=path.read_text(encoding='utf-8-sig');header,first,rest=text.split('\n',2)
            original=next(csv.DictReader(io.StringIO(header+'\n'+first+'\n')))
            variants=[header+'\n'+rest, text+first+'\n']
            for key,value in [('target_end_of_day_storage_m3','nan'),
                              ('zero_lag_cumulative_natural_inflow_m3s','0'),
                              ('reference_target_feasible_zero_lag','2')]:
                row=dict(original);row[key]=value;stream=io.StringIO()
                w=csv.DictWriter(stream,fieldnames=list(row));w.writeheader();w.writerow(row)
                variants.append(stream.getvalue()+rest)
            for i,value in enumerate(variants):
                path.write_text(value,encoding='utf-8')
                d=CascadeData(root,self.physical,self.cfg['experiment'])
                with self.subTest(variant=i),self.assertRaises(ValueError):
                    d.load_seasonal_operation()
                self.assertIsNone(d.seasonal_operation)
                self.assertFalse(any(name in d.hashes for name in SEASONAL_FILES))

    def test_inconsistent_period_and_node_tables_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);d=self.temporary_data(root)
            changes=[(SEASONAL_FILES[1],'period_days_leap_year','151'),
                     (SEASONAL_FILES[1],'weight_generation','0.9'),
                     (SEASONAL_FILES[1],'max_total_release_m3s','1'),
                     (SEASONAL_FILES[2],'node_sequence','2'),
                     (SEASONAL_FILES[2],'reference_target_storage_m3','0')]
            for name,key,value in changes:
                with (root/name).open(encoding='utf-8-sig',newline='') as f:
                    original=list(csv.DictReader(f))
                rows=copy.deepcopy(original);rows[0][key]=value
                with (root/name).open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                with self.subTest(file=name,key=key),self.assertRaises(ValueError):
                    d.load_seasonal_operation()
                shutil.copy2(DATA/name,root/name)

    def test_joint_hard_constraints_are_feasible_for_all_60_years(self):
        d,s=self.data,self.seasonal
        for year in range(1965,2025):
            a,b=d.indices(year,year);low,high=s.storage_bounds(a,b);target=d.initial_storage(year)
            bounds=TerminalReleaseBounds(d,DT,target,d.external[a:b],target,low,high)
            self.assertIsNone(bounds.fast_bounds)
            result=bounds._solve(np.zeros(2*bounds.n))
            self.assertTrue(result.success,'{}: {}'.format(year,result.message))
            q=result.x[:bounds.n].reshape(-1,5)
            end=result.x[bounds.n:].reshape(-1,5)*DT+target
            start=np.vstack((target,end[:-1]));inflow=d.external[a:b].copy();inflow[:,1:]+=q[:,:-1]
            np.testing.assert_allclose(end-start,(inflow-q)*DT,rtol=0.,atol=1e-4)
            self.assertTrue((end>=low-1e-4).all());self.assertTrue((end<=high+1e-4).all())
            self.assertTrue((q>=-1e-8).all());self.assertTrue((q<=d.release_cap+1e-8).all())
            np.testing.assert_allclose(end[-1],target,rtol=0.,atol=1e-4)

    def test_2006_soft_bands_are_not_silently_made_mandatory(self):
        d,s=self.data,self.seasonal;a,b=d.indices(2006,2006);low,high=s.storage_bounds(a,b)
        # Tightening the feasible hard schedule to the suggested daily bands
        # makes the dry-year problem infeasible. The ordinary loader does not.
        for j,id in enumerate(IDS):
            p=d.anchors[id]
            low[:,j]=np.maximum(low[:,j],np.interp(s.reference['soft_target_lower_level_m'][a:b,j],p[:,0],p[:,1]))
            high[:,j]=np.minimum(high[:,j],np.interp(s.reference['soft_target_upper_level_m'][a:b,j],p[:,0],p[:,1]))
        bounds=TerminalReleaseBounds(d,DT,d.initial_storage(2006),d.external[a:b],d.initial_storage(2006),low,high)
        with self.assertRaises(InfeasibleCascade):bounds.interval(0,np.zeros(5))


class SeasonalReachabilityTests(unittest.TestCase):
    def data(self):
        return SimpleNamespace(vmin=np.zeros(5),vmax=np.full(5,10.*DT),release_cap=np.full(5,25.))

    def test_upcoming_flood_ceiling_forces_early_drawdown(self):
        d=self.data();d.vmax=np.array([10.,1.,1.,1.,1.])*DT;d.release_cap[:]=5.
        flows=np.zeros((4,5));flows[2:,0]=4.
        low=np.tile(d.vmin,(4,1));high=np.tile(d.vmax,(4,1));high[1,0]=2.*DT
        old=TerminalReleaseBounds(d,DT,d.vmax,flows,d.vmax)
        new=TerminalReleaseBounds(d,DT,d.vmax,flows,d.vmax,low,high)
        self.assertIsNotNone(old.fast_bounds);self.assertIsNone(new.fast_bounds)
        self.assertEqual(old.interval(0,np.zeros(5))[0],0.)
        np.testing.assert_allclose(new.interval(0,np.zeros(5)),[3.,5.],atol=1e-8,rtol=0)

    def test_intermediate_refill_deadline_prevents_premature_release(self):
        d=self.data();storage=np.array([5.,0.,0.,0.,0.])*DT
        flows=np.zeros((3,5));flows[1,0]=3.
        low=np.tile(d.vmin,(3,1));high=np.tile(d.vmax,(3,1));low[1,0]=8.*DT
        old=TerminalReleaseBounds(d,DT,storage,flows,storage)
        new=TerminalReleaseBounds(d,DT,storage,flows,storage,low,high)
        self.assertAlmostEqual(old.interval(0,np.zeros(5))[1],3.)
        np.testing.assert_allclose(new.interval(0,np.zeros(5)),[0.,0.],atol=1e-8,rtol=0)

    def test_daily_bounds_work_without_a_terminal_target(self):
        d=self.data();d.release_cap[:]=5.
        storage=np.array([10.,0.,0.,0.,0.])*DT;flows=np.zeros((2,5))
        low=np.tile(d.vmin,(2,1));high=np.tile(d.vmax,(2,1));high[1,0]=2.*DT
        bounds=TerminalReleaseBounds(d,DT,storage,flows,None,low,high)
        np.testing.assert_allclose(bounds.interval(0,np.zeros(5)),[3.,5.],atol=1e-8,rtol=0)
        high[0,0]=0.;high[1,0]=0.
        bad=TerminalReleaseBounds(d,DT,storage,flows,None,low,high)
        with self.assertRaises(InfeasibleCascade):bad.interval(0,np.zeros(5))

    def test_terminal_tolerance_never_relaxes_a_seasonal_bound(self):
        d=self.data();d.vmax[:]=1e12
        storage=d.vmax.copy();storage[0]-=50.  # Within legacy 100 m3 closure tolerance.
        flows=np.zeros((2,5));low=np.tile(d.vmin,(2,1));high=np.tile(d.vmax,(2,1))
        legacy=TerminalReleaseBounds(d,DT,storage,flows,d.vmax)
        self.assertEqual(legacy.interval(0,np.zeros(5))[0],0.)
        low[-1]=d.vmax
        strict=TerminalReleaseBounds(d,DT,storage,flows,d.vmax,low,high)
        with self.assertRaises(InfeasibleCascade):strict.interval(0,np.zeros(5))

    def test_invalid_daily_bounds_and_conflicting_terminal_target(self):
        d=self.data();flows=np.ones((2,5));storage=d.vmax/2
        low=np.tile(d.vmin,(2,1));high=np.tile(d.vmax,(2,1))
        cases=[(low,None),(low[:1],high),(low,high[:1]),(high+1,high),
               (low-1,high),(low,high+1),(low*np.nan,high)]
        for a,b in cases:
            with self.assertRaises(ValueError):
                TerminalReleaseBounds(d,DT,storage,flows,None,a,b)
        high[-1,0]=0.
        with self.assertRaisesRegex(InfeasibleCascade,'conflicts'):
            TerminalReleaseBounds(d,DT,storage,flows,d.vmax,low,high)

    def test_intervals_match_independent_cumulative_balance_oracle(self):
        rng=np.random.RandomState(924);d=self.data()
        for trial in range(16):
            days=int(rng.randint(2,7));flows=rng.uniform(2.,3.,(days,5));storage=np.full(5,5.*DT)
            witness=rng.uniform(4.,6.,(days,5))*DT
            low=np.maximum(d.vmin,witness-rng.uniform(0.,1.,(days,5))*DT)
            high=np.minimum(d.vmax,witness+rng.uniform(0.,1.,(days,5))*DT)
            low[days//2,trial%5]=high[days//2,trial%5]=witness[days//2,trial%5]
            target=witness[-1] if trial%2 else None
            bounds=TerminalReleaseBounds(d,DT,storage,flows,target,low,high)
            # Eliminate all storage variables to obtain an independent oracle.
            chain=np.eye(5);chain[np.arange(1,5),np.arange(4)]=-1.
            matrix=np.kron(np.tril(np.ones((days,days))),chain)
            accumulated=np.cumsum(flows,axis=0)
            ub=(accumulated+(storage-low)/DT).reshape(-1)
            lb=(accumulated+(storage-high)/DT).reshape(-1)
            q_bounds=list(zip(np.zeros(days*5),np.tile(d.release_cap,days)))
            eq=None if target is None else matrix[-5:]
            rhs=None if target is None else flows.sum(0)+(storage-target)/DT
            releases=np.full(5,np.nan)
            for j in range(5):
                objective=np.zeros(days*5);objective[j]=1.
                expected=[]
                for sign in (1.,-1.):
                    result=linprog(sign*objective,A_ub=np.vstack((matrix,-matrix)),
                        b_ub=np.concatenate((ub,-lb)),A_eq=eq,b_eq=rhs,bounds=q_bounds,method='highs')
                    self.assertTrue(result.success,result.message)
                    expected.append(result.x[j])
                actual=bounds.interval(j,releases)
                np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
                releases[j]=actual[0]+rng.uniform()*(actual[1]-actual[0])
                q_bounds[j]=(releases[j],releases[j])


class SeasonalExecutionTests(unittest.TestCase):
    def fixture(self, mode='initial'):
        """Five-day cascade with a day-2 flood ceiling and day-4 refill goal."""
        from datetime import date, timedelta
        _,config=load_configuration()
        config.update(head_model='fixed',seasonal_constraints=True,terminal_storage_mode=mode)
        d=SimpleNamespace(vmin=np.zeros(5),vmax=np.array([10.,1.,1.,1.,1.])*DT,
            release_cap=np.full(5,5.),turbine_cap=np.full(5,5.),power_cap=np.full(5,10000.),
            head=np.full(5,100.),start_year=1965,end_year=1965,
            dates=tuple(date(1965,1,1)+timedelta(days=t) for t in range(5)),
            external=np.zeros((5,5)),flow_mean=np.zeros(5),flow_std=np.ones(5),
            period_labels=('drawdown',)*5)
        d.external[2:4,0]=4.
        d.indices=lambda a,b:(0,5)
        d.initial_storage=lambda year:d.vmax.copy()
        d.levels=lambda storage:np.asarray(storage)/DT
        lower=np.tile(d.vmin,(5,1));upper=np.tile(d.vmax,(5,1))
        upper[1,0]=2.*DT;lower[3]=d.vmax
        flood=np.zeros((5,5),dtype=bool);flood[1]=True
        refill=np.zeros((5,5),dtype=bool);refill[3]=True
        targets=np.tile(d.vmax,(5,1));targets[:3,0]=np.array([6.,2.,9.])*DT
        seasonal=SimpleNamespace(storage_lower=lower,storage_upper=upper,
            flood_limit_active=flood,refill_deadline=refill,
            reference={'target_end_of_day_storage_m3':targets,
                       'target_end_of_day_level_m':targets/DT},
            key_parameters={name:{'value_status':'test_scenario'} for name in IDS},
            storage_bounds=lambda a,b:(lower[a:b].copy(),upper[a:b].copy()))
        d.load_seasonal_operation=lambda:seasonal
        return d,config,seasonal

    def test_physics_executes_end_of_day_bounds_without_clipping_water(self):
        d,c,s=self.fixture();p=ReservoirPhysics(d,c)
        # Starting above today's ceiling is valid on the drawdown deadline.
        storage=d.vmax.copy();low=s.storage_lower[1];high=s.storage_upper[1]
        before=[a.copy() for a in (storage,low,high)]
        d.release_cap[:]=20.
        row=p.transition(storage,np.zeros(5),np.zeros(5),storage_lower=low,storage_upper=high)
        self.assertAlmostEqual(row['release'][0],8.)
        self.assertAlmostEqual(row['storage_end'][0],2.*DT)
        np.testing.assert_allclose(row['storage_end']-storage,(row['inflow']-row['release'])*DT,rtol=0,atol=1e-8)
        self.assertTrue((row['storage_end']>=low-1e-8).all())
        self.assertTrue((row['storage_end']<=high+1e-8).all())
        for actual,expected in zip((storage,low,high),before):np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(row['storage_lower_m3'],low)
        np.testing.assert_array_equal(row['storage_upper_m3'],high)
        row['storage_upper_m3'][:]=0
        np.testing.assert_array_equal(high,before[2])
        np.testing.assert_allclose(row['reward'],row['energy'].sum()/p.reference_energy)

    def test_current_day_intervals_match_independent_chain_lp(self):
        d,c,_=self.fixture();d.vmax[:]=10.*DT;d.release_cap[:]=25.
        p=ReservoirPhysics(d,c);rng=np.random.RandomState(321)
        chain=np.eye(5);chain[np.arange(1,5),np.arange(4)]=-1.
        for trial in range(20):
            storage=np.full(5,5.*DT);flows=rng.uniform(2.,3.,5)
            witness=rng.uniform(4.,6.,5)*DT
            lower=witness-rng.uniform(0.,1.,5)*DT;upper=witness+rng.uniform(0.,1.,5)*DT
            j=trial%5;lower[j]=upper[j]=witness[j]
            actions=rng.uniform(size=5)
            row=p.transition(storage,flows,actions,storage_lower=lower,storage_upper=upper)
            q_bounds=list(zip(np.zeros(5),d.release_cap))
            for j in range(5):
                obj=np.eye(5)[j];expected=[]
                for sign in (1.,-1.):
                    result=linprog(sign*obj,A_ub=np.vstack((chain,-chain)),
                        b_ub=np.concatenate((flows+(storage-lower)/DT,-flows-(storage-upper)/DT)),
                        bounds=q_bounds,method='highs')
                    self.assertTrue(result.success,result.message);expected.append(result.x[j])
                np.testing.assert_allclose([row['low'][j],row['high'][j]],expected,rtol=0,atol=1e-8)
                q_bounds[j]=(row['release'][j],row['release'][j])

    def test_bad_daily_bounds_and_impossible_deadline_are_rejected(self):
        d,c,s=self.fixture();p=ReservoirPhysics(d,c);low=s.storage_lower[1];high=s.storage_upper[1]
        cases=[(low,None),(low[:4],high),(low,high[:4]),(high+1,high),
               (low-1,high),(low,d.vmax+1),(low*np.nan,high)]
        for a,b in cases:
            with self.assertRaises(ValueError):
                p.transition(d.vmax,np.zeros(5),np.zeros(5),storage_lower=a,storage_upper=b)
        before=d.vmax.copy()
        with self.assertRaises(InfeasibleCascade):
            p.transition(d.vmax,np.zeros(5),np.zeros(5),storage_lower=low,storage_upper=high)
        np.testing.assert_array_equal(d.vmax,before)

    def test_environment_enforces_all_deadlines_and_canonical_forced_actions(self):
        d,c,s=self.fixture();env=CascadeReservoirEnv(d,c)
        self.assertEqual(env.observation_space[0].shape,(43,))
        for day in range(5):
            obs,state,reward,done,infos,_=env.step(np.full(5,199))
            r=env.last_transition
            self.assertTrue((env.storage>=s.storage_lower[day]-1e-7).all())
            self.assertTrue((env.storage<=s.storage_upper[day]+1e-7).all())
            np.testing.assert_allclose(r['storage_end']-r['storage_start'],(r['inflow']-r['release'])*DT,rtol=0,atol=1e-7)
            self.assertTrue(np.isfinite(obs).all());np.testing.assert_array_equal(obs,state)
            self.assertEqual(infos[0]['seasonal_phase'],r['seasonal_phases'][0])
            self.assertEqual(infos[0]['flood_limit_active'],bool(s.flood_limit_active[day,0]))
            if day==0:self.assertGreaterEqual(r['low'][0],3.-1e-8)
            if day==1:self.assertEqual(infos[0]['seasonal_phase'],'drawdown')
            if day==2:self.assertEqual(infos[0]['seasonal_phase'],'impoundment')
            if day==4:self.assertEqual(infos[0]['seasonal_phase'],'post_refill')
        np.testing.assert_allclose(env.storage,d.vmax,rtol=0,atol=1e-7)
        np.testing.assert_array_equal(env.last_transition['controls'],np.zeros(5))
        np.testing.assert_array_equal(env.last_transition['actions'],np.zeros(5))
        np.testing.assert_array_equal(obs,np.zeros((5,43)))
        self.assertTrue(done.all())
        env.reset();self.assertIsNone(env.last_transition);self.assertEqual(env.index,env.start)

    def test_observations_expose_rules_without_future_inflows(self):
        d,c,_=self.fixture();env=CascadeReservoirEnv(d,c);obs=env.observation()
        np.testing.assert_array_equal(obs[18:23],np.zeros(5))
        np.testing.assert_array_equal(obs[23:28],np.ones(5))
        np.testing.assert_allclose(obs[28:33],np.full(5,2./366),rtol=0,atol=1e-8)
        self.assertAlmostEqual(float(obs[38]),.2,places=7)
        # Future flow changes do not enter the state vector or any rule feature.
        d.external[2:]+=1.
        np.testing.assert_array_equal(env.observation(),obs)
        d.external[2:]-=1.
        # At day 3 the next event is the day-4 refill target.
        env.step(np.zeros(5));env.step(np.zeros(5));obs=env.observation()
        np.testing.assert_array_equal(obs[33:38],np.ones(5))
        np.testing.assert_array_equal(obs[38:43],np.ones(5))

    def test_free_terminal_mode_still_protects_seasonal_deadlines(self):
        d,c,_=self.fixture(mode='free');env=CascadeReservoirEnv(d,c)
        self.assertIsNone(env.terminal_storage_target)
        for _ in range(4):env.step(np.zeros(5))
        np.testing.assert_allclose(env.storage,d.vmax,rtol=0,atol=1e-7)
        np.testing.assert_array_equal(env.observation()[28:33],np.full(5,-1.))
        env.step(np.full(5,199))
        self.assertLess(env.storage.sum(),d.vmax.sum())

    def test_failed_step_and_failed_reset_do_not_mutate_state(self):
        d,c,_=self.fixture();d.external[:]=0.
        env=CascadeReservoirEnv(d,c);before=env.storage.copy()
        with self.assertRaisesRegex(InfeasibleCascade,'date=1965-01-01.*actions='):
            env.step(np.zeros(5))
        np.testing.assert_array_equal(env.storage,before)
        self.assertEqual(env.index,env.start);self.assertIsNone(env.last_transition)
        d.initial_storage=lambda year:np.full(5,np.nan)
        with self.assertRaises(InfeasibleCascade):env.reset()
        np.testing.assert_array_equal(env.storage,before)
        self.assertEqual(env.index,env.start)

    def test_previous_day_violation_is_not_repaired_silently(self):
        d,c,_=self.fixture();env=CascadeReservoirEnv(d,c)
        env.step(np.zeros(5));env.step(np.zeros(5))
        env.storage[0]=3.*DT  # Yesterday's flood ceiling was 2 units.
        before=env.storage.copy();index=env.index;last=env.last_transition
        with self.assertRaisesRegex(InfeasibleCascade,'previous-day'):
            env.step(np.zeros(5))
        np.testing.assert_array_equal(env.storage,before)
        self.assertEqual(env.index,index);self.assertIs(env.last_transition,last)

    def test_loading_alone_does_not_enable_seasonal_execution(self):
        d,c,_=self.fixture();c['seasonal_constraints']=False
        env=CascadeReservoirEnv(d,c);env.step(np.zeros(5))
        self.assertEqual(env.observation().shape,(18,))
        np.testing.assert_array_equal(env.storage,d.vmax)
        self.assertNotIn('storage_upper_m3',env.last_transition)
        with self.assertRaises(ValueError):CascadeReservoirEnv(d,dict(c,seasonal_constraints='true'))

    def test_happo_actors_critic_and_buffers_accept_seasonal_observations(self):
        from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
        d,c,_=self.fixture();d.training_years=(1965,)
        cfg,_=load_configuration()
        runner=CascadeReservoirRunner(d,c,cfg,seed=17)
        runner.collect_episode(1965)
        self.assertEqual(runner.actor_buffer[0].obs.shape,(6,1,43))
        self.assertEqual(runner.critic_buffer.share_obs.shape,(6,1,43))
        self.assertTrue(np.isfinite(runner.critic_buffer.returns).all())
        np.testing.assert_allclose(runner.envs[0].storage,d.vmax,rtol=0,atol=1e-7)
        runner.prep_training();actors,critic=runner.train()
        for metrics in actors+[critic]:
            self.assertTrue(all(np.isfinite(float(v)) for v in metrics.values()))


class SeasonalEvaluationTests(unittest.TestCase):
    def test_reference_is_feasible_and_uses_replayable_policy_actions(self):
        import torch
        from harl.envs.cascade_reservoir.cascade_reservoir_env import SEASONAL_PHASES
        for bins in (2,17,200):
            with self.subTest(bins=bins):
                d,c,s=SeasonalExecutionTests().fixture()
                d.validation_years=(1965,);d.test_years=()
                reference=reference_summary(d,ReservoirPhysics(d,c),[1965],action_bins=bins)[0]
                env=CascadeReservoirEnv(d,c,action_bins=bins);actions=[]
                while env.index<env.stop:
                    env.step_towards_storage(s.reference['target_end_of_day_storage_m3'][env.index])
                    actions.append(env.last_transition['actions'].copy())
                    np.testing.assert_allclose(env.last_transition['normalized_actions'][
                        env.last_transition['controls']>0],
                        env.last_transition['actions'][env.last_transition['controls']>0]/(bins-1),rtol=0,atol=0)
                class ReplayActor:
                    def __init__(self, agent):
                        self.args={'action_bins':bins};self.day=0;self.agent=agent
                    def prep_rollout(self):pass
                    def act(self, obs, deterministic):
                        action=actions[self.day][self.agent];self.day+=1
                        return torch.tensor([[action]])
                row=evaluate_policy([ReplayActor(i) for i in range(5)],d,c,1965,1,0,0,diagnostics=True)
                self.assertAlmostEqual(row['energy_TWh'],reference['energy_TWh'],places=12)
                self.assertEqual(row['violation_count'],0);self.assertTrue(row['feasible'])
                self.assertEqual(reference['reference_type'],'seasonal_target_tracking')
                self.assertEqual(reference['action_bins'],bins)
                self.assertEqual(reference['operation_data_status'],{name:'test_scenario' for name in IDS})
                self.assertEqual(reference['refill_deadline_checks_by_reservoir'],[1]*5)
                self.assertLess(max(reference['refill_deadline_shortfall_m3_by_reservoir']),1e-7)
                self.assertLess(max(reference['flood_ceiling_excess_m3_by_reservoir']),1e-7)
                self.assertLess(max(reference['terminal_storage_error_m3_by_reservoir']),1e-7)
                self.assertGreater(reference['soft_reference_level_mae_m_by_reservoir'][0],0.)
                days=np.sum([row['phase_days_by_reservoir'][p] for p in SEASONAL_PHASES],axis=0)
                np.testing.assert_array_equal(days,np.full(5,5))
                phases=np.sum([row['phase_energy_by_reservoir_TWh'][p] for p in SEASONAL_PHASES],axis=0)
                np.testing.assert_allclose(phases,row['energy_by_reservoir_TWh'],rtol=1e-14)
                for key in ('phase_days_by_reservoir','phase_energy_by_reservoir_TWh',
                            'refill_deadline_shortfall_m3_by_reservoir','soft_reference_level_mae_m_by_reservoir'):
                    self.assertEqual(row[key],reference[key])

    def test_reference_target_validation_preserves_environment_state(self):
        d,c,_=SeasonalExecutionTests().fixture();env=CascadeReservoirEnv(d,c)
        initial=env.storage.copy()
        for target in (np.zeros(4),np.full(5,np.nan),d.vmin-1,d.vmax+1):
            with self.assertRaises(ValueError):env.step_towards_storage(target)
            np.testing.assert_array_equal(env.storage,initial)
            self.assertEqual(env.index,env.start);self.assertIsNone(env.last_transition)

    def test_preflight_rejects_unreachable_rules_with_year_context(self):
        d,c,_=SeasonalExecutionTests().fixture()
        self.assertEqual(validate_operating_feasibility(d,c),[1965])
        d.external[:]=0.
        with self.assertRaisesRegex(InfeasibleCascade,'year 1965.*unreachable'):
            validate_operating_feasibility(d,c)

    def test_reference_default_action_grid_can_be_overridden(self):
        d,c,_=SeasonalExecutionTests().fixture();p=ReservoirPhysics(d,c)
        with self.assertRaisesRegex(ValueError,'action_bins'):
            reference_summary(d,p,[1965],action_bins=1)

    def test_compliance_accounting_detects_a_corrupted_transition(self):
        d,c,s=SeasonalExecutionTests().fixture();env=CascadeReservoirEnv(d,c)
        while env.index<env.stop:
            env.step_towards_storage(s.reference['target_end_of_day_storage_m3'][env.index])
        metrics=SeasonalMetrics(env)
        # Independent reporting must detect a bad record even if the simulation
        # would normally reject it before evaluation sees it.
        env.last_transition['storage_end'][0]=env.last_transition['storage_lower_m3'][0]-DT
        metrics.update()
        self.assertEqual(metrics.violations,1)
        with self.assertRaisesRegex(InfeasibleCascade,'Evaluation trajectory'):
            metrics.finish()


class RealSeasonalExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.physical=load_configuration()
        cls.data=CascadeData(DATA,cls.physical,cls.cfg['experiment'])
        cls.config=dict(cls.physical,seasonal_constraints=True)

    def test_reservoir_specific_phases_and_leap_deadlines(self):
        env=CascadeReservoirEnv(self.data,self.config,start_year=2024)
        self.assertEqual(env.stop-env.start,366)
        # BHT has its own August refill period within the common flood season.
        t=next(i for i,d in enumerate(self.data.dates) if d.isoformat()=='2024-08-01')
        self.assertEqual(env._phase_codes[t-env.start,1],2)
        self.assertEqual(self.data.period_labels[t],'flood_season')
        # The THR deadline is June 10, inclusive at end of day.
        t=next(i for i,d in enumerate(self.data.dates) if d.isoformat()=='2024-06-10')
        self.assertEqual(env._phase_codes[t-env.start,4],0)
        self.assertAlmostEqual(self.data.levels(env.storage_upper[t-env.start])[4],145.)
        self.assertEqual(env._phase_codes[t+1-env.start,4],1)

    def test_free_multi_year_calendar_retains_reservoir_specific_events(self):
        env=CascadeReservoirEnv(self.data,dict(self.config,terminal_storage_mode='free'),start_year=2023,end_year=2024)
        self.assertEqual(env.stop-env.start,731)
        self.assertEqual(env.storage_upper.shape,(731,5))
        self.assertEqual(len(env._deadline_indices[0]),4)
        self.assertIsNone(env.terminal_storage_target)

    def test_full_leap_year_minimum_actions_preserve_constraints_and_closure(self):
        env=CascadeReservoirEnv(self.data,self.config,start_year=2024)
        initial=env.storage.copy()
        while env.index<env.stop:
            offset=env.index-env.start;env.step(np.zeros(5));r=env.last_transition
            np.testing.assert_allclose(r['storage_end']-r['storage_start'],(r['inflow']-r['release'])*DT,rtol=0,atol=1e-4)
            self.assertTrue((env.storage>=env.storage_lower[offset]-1e-3).all())
            self.assertTrue((env.storage<=env.storage_upper[offset]+1e-3).all())
            self.assertTrue((r['release']>=0).all())
            self.assertTrue((r['release']<=self.data.release_cap+1e-8).all())
            self.assertTrue((r['power']<=self.data.power_cap+1e-7).all())
        np.testing.assert_allclose(env.storage,initial,rtol=0,atol=1e-3)


if __name__ == '__main__':
    unittest.main()
