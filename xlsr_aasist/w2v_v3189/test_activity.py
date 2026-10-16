"""Behavioral tests: pauses, recovery, weak boundaries and paired valid regions."""
import unittest
import numpy as np
from .activity import process, settings, mask_pairs


def speech_fixture():
    rng=np.random.default_rng(8);t=np.arange(48000)/16000
    envelope=np.full(len(t),.18)
    envelope[12000:15200]=0.       # 200 ms pause
    envelope[26000:27280]=0.       # 80 ms pause
    envelope[32000:33600]=.0006    # weak 100 ms boundary near the noise floor
    envelope[40000:]=0.           # tail silence, with a continuous noise bed
    return (envelope*np.sin(2*np.pi*240*t)+rng.normal(0,.0007,len(t))).astype(np.float32)


class ActivityTests(unittest.TestCase):
    def test_pause_attenuation_weak_boundary_and_recovery(self):
        x=speech_fixture();cfg=settings(1,'strong')
        cfg.update(hold_ms=10.,release_ms=8.,attack_ms=8.,floor_db=-36.)
        y,m=process(x,cfg)
        def gain(a,b):return 10*np.log10(np.sum(y[a:b]**2)/np.sum(x[a:b]**2))
        self.assertLess(gain(13500,14800),-25.)
        self.assertLess(gain(26700,27100),-12.)
        self.assertLess(gain(33200,33500),-12.)
        self.assertGreater(gain(16000,18000),-.2)
        self.assertEqual(y.shape,x.shape)
        self.assertEqual(y.dtype,np.float32)
        self.assertGreater(m['gate_closures'],1)
        self.assertGreater(m['gate_openings'],1)
        self.assertTrue(any(a<=14000<b for a,b in m['suppressed_spans']))
        self.assertFalse(np.any(y==0))  # finite attenuation, not arbitrary zero insertion

    def test_hold_protects_brief_dips_and_envelope_is_smooth(self):
        x=speech_fixture();short=settings(2,'strong');short.update(hold_ms=0.,release_ms=10.)
        long=dict(short,hold_ms=120.)
        a,_=process(x,short);b,_=process(x,long)
        self.assertLess(np.mean(a[26400:27100]**2),np.mean(b[26400:27100]**2))
        # On nonzero samples, attenuation gains cannot abruptly switch state.
        valid=np.abs(x)>.00001;g=np.zeros(len(x));g[valid]=a[valid]/x[valid]
        both=valid[:-1]&valid[1:]
        self.assertLess(np.abs(np.diff(g))[both].max(),.03)

    def test_disabled_silence_flat_and_tiny_inputs_are_not_destroyed(self):
        for x in (np.zeros(16000,np.float32),np.ones(16000,np.float32)*.1,np.array([.2],np.float32)):
            for profile in ('off','mild','strong'):
                y,m=process(x,settings(5,profile))
                np.testing.assert_array_equal(y,x)
                self.assertEqual(m['suppressed_spans'],[])
        x=speech_fixture();y,_=process(x,settings(3,'off'));np.testing.assert_array_equal(x,y)
        with self.assertRaises(ValueError):process(np.array([np.nan],np.float32),settings(1))

    def test_repeatability_no_label_input_and_mixed_profile_coverage(self):
        x=speech_fixture()
        first,m=process(x,settings(7));again,n=process(x,settings(7))
        np.testing.assert_array_equal(first,again);self.assertEqual(m,n)
        counts={p:0 for p in ('off','mild','strong')}
        for seed in range(1000):counts[settings(seed)['profile']]+=1
        self.assertTrue(190<counts['off']<310)
        self.assertTrue(380<counts['mild']<520)
        self.assertTrue(230<counts['strong']<370)

    def test_mask_both_ends_union_with_different_phone_durations(self):
        pairs=[[0,4000,1000,9000]]
        safe,affected=mask_pairs(pairs,[[1000,1500]],[[6000,7000]],guard=0)
        self.assertEqual(affected,1)
        self.assertEqual(safe,[[0.,1000.,1000.,3000.],[1500.,2500.,4000.,6000.],[3000.,4000.,7000.,9000.]])
        for a,b,c,d in safe:self.assertAlmostEqual((b-a)*2,d-c)
        self.assertEqual(mask_pairs(pairs,[[0,4000]],[])[0],[])
        self.assertEqual(mask_pairs(pairs,[],[])[0],pairs)

    def test_mask_guard_rejects_receptive_fields_at_suppressed_boundary(self):
        safe,affected=mask_pairs([[0,4000,0,4000]],[],[[1500,2500]])
        self.assertEqual(safe,[[0.,1300.,0.,1300.],[2700.,4000.,2700.,4000.]])
        self.assertEqual(affected,1)
        self.assertEqual(mask_pairs([[0,300,0,300]],[],[[250,300]])[0],[[0.,50.,0.,50.]])


if __name__=='__main__':unittest.main()
