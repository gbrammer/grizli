"""
Helper function for running Grizli redshift fits in AWS lambda

event = {'s3_object_path' : 'Pipeline/j001452+091221/Extractions/j001452+091221_00277.beams.fits'}

Optional event keys:

    'verbose' : verbose output
    
    'skip_started' : Look for a start.log file and abort if found
    
    'check_wcs' : check for WCS files, needed for ACS fits
    
    'quasar_fit' : run fit with quasar templates
    
    'use_psf' : Use point source models (for quasar fits)
    
    'output_path' : optional output path in the aws-grivam bucket
    
    'zr' : [zmin, zmax] Redshift fitting range
    
"""

import os
import glob
import time

import numpy as np

import boto3

def extract_beams_from_flt(root, bucket, id):
    """
    Download GrismFLT files and extract the beams file
    """
    import grizli
    from grizli import fitting, utils, multifit
    utils.set_warnings()
    from grizli.pipeline import auto_script
    
    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bkt = s3.Bucket(bucket)
    
    # WCS files for ACS
    files = [obj.key for obj in bkt.objects.filter(Prefix='Pipeline/{0}/Extractions/j'.format(root))]
    files += [obj.key for obj in bkt.objects.filter(Prefix='Pipeline/{0}/Extractions/i'.format(root))]

    files += [obj.key for obj in bkt.objects.filter(Prefix='Pipeline/{0}/Extractions/{0}-ir.cat.fits'.format(root))]

    files += [obj.key for obj in bkt.objects.filter(Prefix='Pipeline/{0}/Extractions/fit_args.npy'.format(root))]
    
    download_files = []
    for file in np.unique(files):
        if ('cat.fits' in file) | ('fit_args' in file):
            if os.path.exists(os.path.basename(file)):
                continue
            
            download_files.append(file)
        
    for file in download_files:
        print(file)    
        bkt.download_file(file, os.path.basename(file),
                          ExtraArgs={"RequestPayer": "requester"})
    
    # One beam at a time
    beams = None
    
    flt_files = []
    for file in files:
        if 'GrismFLT.fits' in file:
            flt_files.append(file)
    
    print('Read {0} GrismFLT files'.format(len(flt_files)))
    
    for i, file in enumerate(flt_files):
        print('# Read {0}/{1}'.format(i+1, len(flt_files)))

        flt, ext, _, _ = os.path.basename(file).split('.')          
        if flt.startswith('i'):
            fl = 'flt'
        else:
            fl = 'flc'
        
        out_files = ['{0}.{1}.GrismFLT.fits'.format(flt, ext), 
                     '{0}.{1}.GrismFLT.pkl'.format(flt, ext), 
                     '{0}_{2}.{1}.wcs.fits'.format(flt, ext, fl)]
        
        for f_j in out_files:             
            aws_file = os.path.join(os.path.dirname(file), f_j)
            print('  ', aws_file)
            if not os.path.exists(f_j):
                bkt.download_file(aws_file, f_j, 
                                  ExtraArgs={"RequestPayer": "requester"})
        
        beams_i =                           auto_script.extract(field_root=root, maglim=[13,24], prior=None, MW_EBV=0.00, ids=id, pline={}, fit_only_beams=True, run_fit=False, poly_order=7, master_files=[os.path.basename(file)], grp=None, bad_pa_threshold=None, fit_trace_shift=False, size=32, diff=True, min_sens=0.02, skip_complete=True, fit_args={}, args_file='fit_args.npy', get_only_beams=True)
        if beams is None:
            beams = beams_i
        else:
            beams.extend(beams_i)
        
        # Remove the GrismFLT file    
        for f_j in out_files:
            if 'GrismFLT' in f_j:
                os.remove(f_j)
                
    if beams is None:
        print('No beams found for {0} id={1}'.format(root, id))
        return False
    
    # Grism Object
    args = np.load('fit_args.npy', allow_pickle=True)[0]
    mb = multifit.MultiBeam(beams, **args)
    mb.write_master_fits()
    
    # 1D spectrum with R=30 fit
    if True:
        bin_steps, step_templ = utils.step_templates(wlim=[5000, 18000.0], 
                                                     R=30, round=10)  

        tfit = mb.template_at_z(z=0, templates=step_templ,
                                fit_background=True, fitter='lstsq', 
                                get_uncertainties=2)
        
        fig1 = mb.oned_figure(figsize=[5,3], tfit=tfit, show_beams=True, 
                              scale_on_stacked=True, ylim_percentile=5)
                              
        outroot='{0}_{1:05d}.R{2:.0f}'.format(root, id, 30)
        hdu = mb.oned_spectrum_to_hdu(outputfile=outroot+'.fits', 
                                              tfit=tfit, wave=bin_steps)                     
        fig1.savefig(outroot+'.png')
        
    return('{0}_{1:05d}.beams.fits'.format(root, id))
    
def run_grizli_fit(event):
    import grizli
    from grizli import fitting, utils, multifit
    utils.set_warnings()
    
    #event = {'s3_object_path':'Pipeline/j001452+091221/Extractions/j001452+091221_00277.beams.fits'}
    
    ###
    ### Parse event arguments
    ### 
    event_kwargs = {}
    for k in event:
        # Bool
        if event[k].lower() in ['false', 'true']:
            event_kwargs[k] = event[k].lower() in ["true"]
        
        # None
        if event[k].lower() in ['none']:
            event_kwargs[k] = None
        
        # Lists
        if ',' in event[k]:
            try:
                event_kwargs[k] = np.cast[float](event[k].split(','))
            except:
                event_kwargs[k] = event[k].split(',')
        
    # Defaults
    for k in ['verbose', 'check_wcs', 'quasar_fit', 'use_psf', 'skip_started', 'extract_from_flt']:
        if k not in event_kwargs:
            event_kwargs[k] = False
            
    # for k in ['verbose', 'check_wcs', 'quasar_fit', 'use_psf', 'skip_started', 'extract_from_flt']:
    #     if k in event:
    #         event_bools[k] = event[k].lower() in ["true", True]
    #     else:
    #         event_bools[k] = False
    
    print('Grizli version: ', grizli.__version__)
    
    ## Output path
    if 'output_path' in event:
        output_path = event['output_path']
    else:
        output_path = None
    
    if 'bucket' not in event:
        event_kwargs['bucket'] = event['bucket']
    else:
        event_kwargs['bucket'] = 'aws-grivam'
                        
    os.chdir('/tmp/')
    os.system('cp {0}/matplotlibrc .'.format(grizli.GRIZLI_PATH))
    
    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bkt = s3.Bucket(event_kwargs['bucket'])

    beams_file = os.path.basename(event['s3_object_path'])
    root = beams_file.split('_')[0]
    id = int(beams_file.split('_')[1].split('.')[0])
     
    # Initial log
    start_log = '{0}_{1:05d}.start.log'.format(root, id)
    full_start = 'Pipeline/{0}/Extractions/{1}'.format(root, start_log)
    
    if event_kwargs['skip_started']:
        res = [r.key for r in bkt.objects.filter(Prefix=full_start)]
        if res:
            print('Already started ({0}), aborting.'.format(start_log))
            return True
            
    fp = open(start_log,'w')
    fp.write(time.ctime()+'\n')
    fp.close()
    bkt.upload_file(start_log, full_start)
    
    # Download fit arguments
    aws_args = 'Pipeline/{0}/Extractions/fit_args.npy'.format(root)
    bkt.download_file(aws_args, './fit_args.npy', ExtraArgs={"RequestPayer": "requester"})
    
    # If no beams file in the bucket, try to generate it
    try:
        bkt.download_file(event['s3_object_path'], './{0}'.format(beams_file), ExtraArgs={"RequestPayer": "requester"})
        put_beams = False
    except:
        print('Extract from GrismFLT object!')
        status = extract_beams_from_flt(root, event_kwargs['bucket'], id)
        if status is False:
            return False
        else:
            beams_file = status
            
        put_beams = True
        
        # upload it now
        output_path = 'Pipeline/{0}/Extractions'.format(root)
        aws_file = '{0}/{1}'.format(output_path, beams_file)
        print(aws_file)
        bkt.upload_file(beams_file, aws_file, 
                        ExtraArgs={'ACL': 'public-read'})
        
        if 'run_fit' in event:
            if event['run_fit'].lower() == 'false':
                return True
                
    # Download WCS files
    if event_kwargs['check_wcs']:
        # WCS files for ACS
        files = [obj.key for obj in bkt.objects.filter(Prefix='Pipeline/{0}/Extractions/j'.format(root))]
        for file in files:
            if 'wcs.fits' in file:
                if os.path.exists(os.path.basename(file)):
                    continue
                
                bkt.download_file(file, os.path.basename(file),
                                  ExtraArgs={"RequestPayer": "requester"})
     
    # Is zr in the event dict?
    if 'zr' in event:
        zr = list(np.cast[float](event['zr']))
    else:
        try:
            zr = np.load('fit_args.npy')[0]['zr']
        except:
            zr = np.load('fit_args.npy', allow_pickle=True)[0]['zr']
    
    # Directory listing
    files = glob.glob('*')
    for i, file in enumerate(files):
        print('File ({0}): {1}'.format(i+1, file))
                
    ###   
    ### Run the fit
    
    if event_kwargs['quasar_fit']:
        
        # Quasar templates
        uv_lines = zr[1] > 3.5
        t0, t1 = utils.load_quasar_templates(uv_line_complex=uv_lines,
                                            broad_fwhm=2800, narrow_fwhm=1000,
                                            fixed_narrow_lines=True, 
                                            nspline=13)
        
        fitting.run_all_parallel(id, t0=t0, t1=t1, fit_only_beams=True,
                                 fit_beams=False,  zr=zr, phot_obj=None, 
                                 **event_kwargs)
        
        if output_path is None:
            output_path = 'Pipeline/QuasarFit'.format(root)
        
    else:
        
        # Normal galaxy redshift fit
        fitting.run_all_parallel(id, zr=zr, fit_only_beams=True,
                                 fit_beams=False,  **event_kwargs)
        
        if output_path is None:
            output_path = 'Pipeline/{0}/Extractions'.format(root)
        
    # Output files
    files = glob.glob('{0}_{1:05d}*'.format(root, id))
    for file in files:
        if ('beams.fits' not in file) | put_beams:
            aws_file = '{0}/{1}'.format(output_path, file)
            print(aws_file)
            bkt.upload_file(file, aws_file, ExtraArgs={'ACL': 'public-read'})
    
    # Remove start log now that done
    res = bkt.delete_objects(Delete={'Objects':[{'Key':full_start}]})

def clean(root='', verbose=True):
    import glob
    import os
    files = glob.glob(root+'*')
    files.sort()
    for file in files:
        print('Cleanup: {0}'.format(file))
        os.remove(file)
        
TESTER = 'Pipeline/j001452+091221/Extractions/j001452+091221_00277.beams.fits'
def run_test(s3_object_path=TESTER):
    event = {'s3_object_path': s3_object_path, 'verbose':'True'}
    run_grizli_fit(event)
    
def handler(event, context):
    import traceback
    
    print(event) #['s3_object_path'], event['verbose'])
    print(context)
    #print('log_stream_name: {0}'.format(context.log_stream_name))
    #context.log_stream_name = event['s3_object_path'].replace('/','_')
    
    try:
        run_grizli_fit(event)
    except:
        print(traceback.format_exc(limit=2)) 
        
    # Clean up
    beams_file = os.path.basename(event['s3_object_path'])
    root = beams_file.split('_')[0]
    clean(root=root, verbose=True)
        
    # # Clean out the log
    # beams_file = os.path.basename(event['s3_object_path'])
    # root = beams_file.split('_')[0]
    # id = int(beams_file.split('_')[1].split('.')[0])
    # 
    # # Start log
    # start_log = '{0}_{1:05d}.start.log'.format(root, id)
    # full_start = 'Pipeline/{0}/Extractions/{1}'.format(root, start_log)
    # 
    # print('Failed on {0}, remove {1}.'.format(event['s3_object_path'], 
    #                                           full_start))
    # 
    # if 'bucket' in event:
    #     bucket = event['bucket']
    # else:
    #     bucket = 'aws-grivam'
    # 
    # s3 = boto3.resource('s3')
    # bkt = s3.Bucket(bucket)
    # res = bkt.delete_objects(Delete={'Objects':[{'Key':full_start}]})
        
def show_version(event, context):
    import grizli
    import eazy
    
    print('Event: ', event)
    
    print('grizli version: ', grizli.__version__)
    print('eazy version: ', eazy.__version__)
    
    import matplotlib
    print('matplotlibrc: ', matplotlib.matplotlib_fname())
    
if __name__ == "__main__":
    handler('', '')
