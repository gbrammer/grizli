"""
Tests for JWST imaging and spectra, including 
"""
import os
import glob
import unittest
import numpy as np
import astropy.io.fits as pyfits
import logging
import pytest

from grizli import prep, utils, multifit, GRIZLI_PATH, jwst_utils
from grizli.pipeline import auto_script


def set_crds(path='crds_cache'):
    """
    Set CRDS environment variables if not already set
    """
    if os.getenv('CRDS_SERVER_URL') is None:
        os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'
        logging.warn("Set CRDS_SERVER_URL = 'https://jwst-crds.stsci.edu'")
        
    if os.getenv('CRDS_PATH') is None:
        if not path.startswith('/'):
            path = os.path.join(os.getcwd(), path)
        
        os.environ['CRDS_PATH'] = path
            
        logging.warn(f"Set CRDS_PATH = '{path}'")
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(os.path.join(path, 'config'))
            os.mkdir(os.path.join(path, 'config', 'jwst'))
            
set_crds()

class TestJWSTHeaders:
    
    def get_test_data_path(self):
        path = os.path.dirname(utils.__file__)
        path_to_files = os.path.join(path, 'tests/data/jwst-headers/')
        return path_to_files
        
    def fetch_test_data(self):
        """
        Fetch test data from MAST
        """
        pass
        
    def test_process_headers(self, tmp_path):
        """
        """
        pytest.importorskip('jwst')
                
        file_path = self.get_test_data_path()
        
        files = glob.glob(os.path.join(file_path, '*fits.gz'))
        files.sort()
        
        assert(len(files) == 5)
        
        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        
        # fresh_flt_file
        gz_file = os.path.basename(files[0])
        prep.fresh_flt_file(gz_file, path=file_path)
        
        local_file = gz_file.split('.gz')[0]
        prep.fresh_flt_file(local_file, path=file_path)
        
        # Multiple modes
        for file in files:
            gz_file = os.path.basename(file)
            local_file = gz_file.split('.gz')[0]
            prep.fresh_flt_file(local_file, path=file_path)
            im = pyfits.open(local_file)
            
            assert('SIPRAMAD' in im['SCI'].header)
            assert(im['SCI'].header['SIPRAMAD'] < 1.e-2)
            assert(im['SCI'].header['SIPDEMAD'] < 1.e-2)
            
        # Parse visits
        visits, groups, info = auto_script.parse_visits(field_root='jwst', 
                                                        RAW_PATH=file_path, 
                                                        visit_split_shift=1.2)
        
        assert(os.path.exists('jwst_visits.yaml'))
        
        assert(len(visits) == 5)
        assert(len(groups) == 1)
        
        os.chdir(orig_dir)


class TestJWSTUtils:
    
    def test_filter_info(self):
        """
        Read the info file and get filter data
        """
        import astropy.io.fits as pyfits
        
        bp = jwst_utils.load_jwst_filter_info()
        assert('meta' in bp)
        
        # NIRISS
        header = pyfits.Header()
        header['TELESCOP'] = 'JWST'
        header['INSTRUME'] = 'NIRISS'
        header['PUPIL'] = 'F200W'
        header['FILTER'] = 'CLEAR'
        header['DETECTOR'] = 'NIS'
        
        assert(utils.parse_filter_from_header(header) == 'F200W-CLEAR')
                
        info = jwst_utils.get_jwst_filter_info(header)
        
        assert(info['name'] == 'F200W')
        assert(np.allclose(info['pivot'], 1.992959))
        
        # NIRCam
        header = pyfits.Header()
        header['TELESCOP'] = 'JWST'
        header['INSTRUME'] = 'NIRCAM'
        header['FILTER'] = 'F200W'
        header['PUPIL'] = 'CLEAR'
        header['DETECTOR'] = 'NRCA1'
        
        assert(utils.parse_filter_from_header(header) == 'F200W-CLEAR')
                
        info = jwst_utils.get_jwst_filter_info(header)
        
        assert(info['name'] == 'F200W')
        assert(np.allclose(info['pivot'], 1.988647))
        
        # MIRI
        header = pyfits.Header()
        header['TELESCOP'] = 'JWST'
        header['INSTRUME'] = 'MIRI'
        header['FILTER'] = 'F560W'
        
        assert(utils.parse_filter_from_header(header) == 'F560W')
        
        info = jwst_utils.get_jwst_filter_info(header)
        assert(info['name'] == 'F560W')
        assert(np.allclose(info['pivot'], 5.632612))


class TestJWSTFittingTools:
    
    def test_config(self):
        """
        Fetch config files if CONF not found
        """
        conf_path = os.path.join(GRIZLI_PATH, 'CONF')
        if not os.path.exists(conf_path): # if CONF dir doesn't already exist, create it
            os.mkdir(conf_path)
        
        if not os.path.exists(os.path.join(conf_path,
                              'GR150C.F115W.conf')):
            print(f'Download config and calib files to {conf_path}')
            #utils.fetch_default_calibs(ACS=False)
            utils.fetch_config_files(get_epsf=True, get_jwst=True)
            files = glob.glob(f'{conf_path}/*')
            print('Files: ', '\n'.join(files))

        assert os.path.exists(os.path.join(conf_path,
                              'GR150C.F115W.conf'))
    
    def test_multibeam(self):
        """
        Can we initialize a multibeam file?
        """
        path = os.path.dirname(utils.__file__)
        print(path)
        beams_file = path + '/tests/data/niriss_jw01324001001_test.beams.fits'
        mb = multifit.MultiBeam(beams_file, group_name='jw01324001001',
                                MW_EBV=-1, fcontam=0.1, sys_err=0.03)
    
        assert(mb.N == 2)
        assert('F115W' in mb.PA)
        
        _ = mb.compute_model()
        
        spec = mb.oned_spectrum()
    
    #def test_parse_visits(self):
    #    """
    #    """
    #    
    #    # These are copies of the WFC3-ERSII demo FLT files but where 
    #    # the data extensions have been set to dtype=uint8 so that they 
    #    # gzip up small
    #    
    #    path = os.path.dirname(utils.__file__)        
    #    if not os.path.exists('niriss-parse-test'):
    #        os.system(f'tar xzvf {path}/tests/data/niriss-parse-test.tar.gz')
    #        
    #    os.chdir('jwst-parse-test/Prep/')
    #    
    #    files = glob.glob('../RAW/*rate.fits')
    #    files.sort()
    #    info = utils.get_flt_info(files)
    #    
    #    assert(len(info) == len(files))
    #    
    #    _visits, _filters = utils.parse_flt_files(info=info, 
    #                                  uniquename=True, get_footprint=True, 
    #                                  use_visit=True, max_dt=0.5, 
    #                                  visit_split_shift=2)
    #    
    #    assert(len(_visits) == 4)
    #    
    #    _groups = utils.parse_grism_associations(_visits)
    #    assert(len(_groups) == 2)
    #    
    #    ### From auto_script
    #    kwargs = auto_script.get_yml_parameters(local_file='my_params.yml', 
    #                                            copy_defaults=False)
    #    
    #    root = 'test'
    #    
    #    for bval in [True, False]:
    #        kwargs['parse_visits_args']['combine_same_pa'] = bval
    #        _ = auto_script.parse_visits(field_root=root,
    #                                     **kwargs['parse_visits_args'])
#
    #        visits, groups, info = _
    #    
    #        assert(hasattr(info, 'colnames'))
    #        assert(len(info) == 16)
    #    
    #        assert(len(visits) == 4)
    #    
    #    assert(os.path.exists(f'{root}_visits.yaml'))
    #    
    #    __ = auto_script.load_visits_yaml(f'{root}_visits.yaml')
    #    _visits, _groups, _info = __
    #    
    #    for v in [visits[0], _visits[0]]:
    #        for k in ['product','files','footprint']:
    #            assert(k in v)
    #    
    #        assert(v['product'].startswith('wfc3-ers'))
    #        assert(len(v['files']) == 4)
    #    
    #    # Grism groups
    #    for g in [groups, _groups]:
    #        assert(len(g) == 2)
    #        
    #        gi = g[0]
    #        for k in ['grism', 'direct']:
    #            assert(k in gi)
    #        
    #        gr = gi['grism'] 
    #        assert(gr['product'] == 'wfc3-ersii-g01-b6o-21-119.0-g102')
    #        assert(len(gr['files']) == 4)
    #        
    #    os.chdir('../../')
    #    if os.path.exists('wfc3-parse-test'):
    #        os.system('rm -rf wfc3-parse-test')
        
    