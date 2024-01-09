import numpy as np
import pytest

from act.qc.sp2 import PYSP2_AVAILABLE, SP2ParticleCriteria


@pytest.mark.skipif(not PYSP2_AVAILABLE, reason='PySP2 is not installed.')
def test_sp2_particle_config():
    particle_config_ds = SP2ParticleCriteria()
    assert particle_config_ds.ScatMaxPeakHt1 == 60000
    assert particle_config_ds.ScatMinPeakHt1 == 250
    assert particle_config_ds.ScatMaxPeakHt2 == 60000
    assert particle_config_ds.ScatMinPeakHt2 == 250
    assert particle_config_ds.ScatMinWidth == 10
    assert particle_config_ds.ScatMaxWidth == 90
    assert particle_config_ds.ScatMinPeakPos == 20
    assert particle_config_ds.ScatMaxPeakPos == 90
    assert particle_config_ds.IncanMinPeakHt1 == 200
    assert particle_config_ds.IncanMinPeakHt2 == 200
    assert particle_config_ds.IncanMaxPeakHt1 == 60000
    assert particle_config_ds.IncanMaxPeakHt2 == 60000
    assert particle_config_ds.IncanMinWidth == 5
    assert particle_config_ds.IncanMaxWidth == np.inf
    assert particle_config_ds.IncanMinPeakPos == 20
    assert particle_config_ds.IncanMaxPeakPos == 90
    assert particle_config_ds.IncanMinPeakRatio == 0.1
    assert particle_config_ds.IncanMaxPeakRatio == 25
    assert particle_config_ds.IncanMaxPeakOffset == 11
    assert particle_config_ds.c0Mass1 == 0
    assert particle_config_ds.c1Mass1 == 0.0001896
    assert particle_config_ds.c2Mass1 == 0
    assert particle_config_ds.c3Mass1 == 0
    assert particle_config_ds.c0Mass2 == 0
    assert particle_config_ds.c1Mass2 == 0.0016815
    assert particle_config_ds.c2Mass2 == 0
    assert particle_config_ds.c3Mass2 == 0
    assert particle_config_ds.c0Scat1 == 0
    assert particle_config_ds.c1Scat1 == 78.141
    assert particle_config_ds.c2Scat1 == 0
    assert particle_config_ds.c0Scat2 == 0
    assert particle_config_ds.c1Scat2 == 752.53
    assert particle_config_ds.c2Scat2 == 0
    assert particle_config_ds.densitySO4 == 1.8
    assert particle_config_ds.densityBC == 1.8
    assert particle_config_ds.TempSTP == 273.15
    assert particle_config_ds.PressSTP == 1013.25
