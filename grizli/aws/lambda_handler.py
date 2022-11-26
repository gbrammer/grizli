"""
Helper function for running Grizli redshift fits in AWS lambda

event = {'s3_object_path' : 'Pipeline/j001452+091221/Extractions/j001452+091221_00277.beams.fits'}

Optional event key (and anything to grizli.fitting.run_all):

    'skip_started' : Look for a start.log file and abort if found

    'check_wcs' : check for WCS files, needed for ACS fits

    'quasar_fit' : run fit with quasar templates

    'output_path' : optional output path in the aws-grivam bucket

Example run_all arguments:

    'use_psf' : Use point source models (for quasar fits)

    'verbose' : verbose output

    'zr' : [zmin, zmax] Redshift fitting range

"""

import os
import glob
import time

import numpy as np

FALSE_OPTIONS = [False, 'false', 'False', 0, '0', 'n', 'N']
TRUE_OPTIONS = [True,  'true',  'True',  1, '1', 'y', 'Y']


def check_object_in_footprint(id, wcs_fits, cat, rd=None):
    """
    Check that a given object is within a WCS footprint

    Parameters
    ----------
    id : int
        Integer id of the catalog objects

    wcs_fits : str
        WCS filename, like 'ibfuw1psq_flt.01.wcs.fits'.

    cat : `~astropy.table.Table`
        Table object of the "ir_cat.fits" source detection table, with
        SExtractor columns NUMBER, X_WORLD, Y_WORLD.

    """
    import matplotlib.path
    import astropy.wcs as pywcs
    import astropy.io.fits as pyfits

    if rd is None:
        ix = cat['NUMBER'] == id
        ra, dec = cat['X_WORLD'][ix][0], cat['Y_WORLD'][ix][0]
    else:
        ra, dec = rd

    im = pyfits.open(wcs_fits)
    im[0].header['NAXIS'] = 2
    im[0].header['NAXIS1'] = im[0].header['CRPIX1']*2
    im[0].header['NAXIS2'] = im[0].header['CRPIX2']*2

    wcs = pywcs.WCS(im[0].header, fobj=im, relax=True)
    fp = matplotlib.path.Path(wcs.calc_footprint())
    has_point = fp.contains_point([ra, dec])
    im.close()

    return has_point


def extract_beams_from_flt(root, bucket, id, clean=True, silent=False):
    """
    Download GrismFLT files and extract the beams file
    """
    import gc
    import boto3

    import matplotlib.pyplot as plt

    import grizli
    from grizli import fitting, utils, multifit
    #from grizli import __version__ as grizli__version
    grizli__version = grizli.__version__
    
    utils.set_warnings()
    from grizli.pipeline import auto_script

    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bkt = s3.Bucket(bucket)

    # WCS files for ACS
    files = [obj.key for obj in 
             bkt.objects.filter(Prefix=f'HST/Pipeline/{root}/Extractions/j')]
    files += [obj.key for obj in 
              bkt.objects.filter(Prefix=f'HST/Pipeline/{root}/Extractions/i')]

    prefix = f'HST/Pipeline/{root}/Extractions/{root}-ir.cat.fits'
    files += [obj.key for obj in bkt.objects.filter(Prefix=prefix)]

    prefix = f'HST/Pipeline/{root}/Extractions/fit_args.npy'
    files += [obj.key for obj in bkt.objects.filter(Prefix=prefix)]

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

    # Read the catalog
    ircat = utils.read_catalog('{0}-ir.cat.fits'.format(root))
    ix = ircat['NUMBER'] == id
    object_rd = (ircat['X_WORLD'][ix], ircat['Y_WORLD'][ix])
    del(ircat)

    # One beam at a time
    beams = None

    flt_files = []
    for file in files:
        if 'GrismFLT.fits' in file:
            flt_files.append(file)

    if not silent:
        print('Read {0} GrismFLT files'.format(len(flt_files)))
    
    if os.path.exists('{0}_fit_args.npy'.format(root)):
        args_file = '{0}_fit_args.npy'.format(root)
    else:
        args_file = 'fit_args.npy'

    for i, file in enumerate(flt_files):
        if not silent:
            print('# Read {0}/{1}'.format(i+1, len(flt_files)))

        flt, ext, _, _ = os.path.basename(file).split('.')
        if flt.startswith('i'):
            fl = 'flt'
        elif flt.startswith('jw'):
            fl = 'rate'
        else:
            fl = 'flc'

        out_files = ['{0}_{2}.{1}.wcs.fits'.format(flt, ext, fl),
                     '{0}.{1}.GrismFLT.fits'.format(flt, ext),
                     '{0}.{1}.GrismFLT.pkl'.format(flt, ext)]

        exp_has_id = False

        for j, f_j in enumerate(out_files):
            aws_file = os.path.join(os.path.dirname(file), f_j)
            if not silent:
                print('  ', aws_file)

            if not os.path.exists(f_j):
                bkt.download_file(aws_file, f_j,
                                  ExtraArgs={"RequestPayer": "requester"})

            # WCS file, check if object in footprint
            if f_j.endswith('.wcs.fits'):
                exp_has_id = check_object_in_footprint(None, f_j, None,
                                                       rd=object_rd)
                if not exp_has_id:
                    if clean:
                        os.remove(f_j)
                    break

        if not exp_has_id:
            continue

        beams_i = auto_script.extract(field_root=root, maglim=[13, 24],
                                      prior=None, MW_EBV=0.00, ids=id,
                                      pline={}, fit_only_beams=True,
                                      run_fit=False, poly_order=7,
                                      master_files=[os.path.basename(file)],
                                      grp=None, bad_pa_threshold=None,
                                      fit_trace_shift=False, size=32,
                                      diff=True, min_sens=0.02,
                                      skip_complete=True, fit_args={},
                                      args_file=args_file,
                                      get_only_beams=True)

        # Remove the GrismFLT file
        for f_j in out_files:
            if ('GrismFLT' in f_j) & clean:
                os.remove(f_j)

        if beams is None:
            beams = beams_i
        else:
            beams.extend(beams_i)

    # Garbage collector
    gc.collect()

    if not beams:
        print('No beams found for {0} id={1}'.format(root, id))
        return False

    # Grism Object
    args = np.load(args_file, allow_pickle=True)[0]
    if '_ehn_' in root:
        try:
            import wfc3dash.grism.grism
            mb = wfc3dash.grism.grism.run_align_dash(beams, args)
        except ImportError:
            mb = multifit.MultiBeam(beams, **args)
    else:
        mb = multifit.MultiBeam(beams, **args)
        
    mb.write_master_fits()

    # 1D spectrum with R=30 fit
    if True:
        bin_steps, step_templ = utils.step_templates(wlim=[5000, 18000.0],
                                                     R=30, round=10)

        tfit = mb.template_at_z(z=0, templates=step_templ,
                                fit_background=True, fitter='lstsq',
                                get_uncertainties=2)

        fig1 = mb.oned_figure(figsize=[5, 3], tfit=tfit, show_beams=True,
                              scale_on_stacked=True, ylim_percentile=5)

        outroot = '{0}_{1:05d}.R{2:.0f}'.format(root, id, 30)
        hdu = mb.oned_spectrum_to_hdu(outputfile=outroot+'.fits',
                                              tfit=tfit, wave=bin_steps)

        fig1.savefig(outroot+'.png')
        del(hdu)

        # Drizzled spectrum
        hdu, fig = mb.drizzle_grisms_and_PAs(fcontam=args['fcontam'],
                                             flambda=False,
                                             kernel='point', size=32,
                                             tfit=tfit, diff=False)

        hdu[0].header['GRIZLIV'] = (grizli__version, 'Grizli version')

        fig.savefig('{0}_{1:05d}.stack.png'.format(root, id))

        hdu.writeto('{0}_{1:05d}.stack.fits'.format(root, id),
                    overwrite=True)

        plt.close('all')
        del(hdu)

    outfiles = ['{0}_{1:05d}.beams.fits'.format(root, id)]
    outfiles += glob.glob(outroot+'*')
    outfiles += glob.glob('{0}_{1:05d}.stack*'.format(root, id))

    return(outfiles)


def run_grizli_fit(event):
    import boto3
    import json
    import shutil
    import gc

    import matplotlib.pyplot as plt

    import grizli
    from grizli import fitting, utils, multifit

    try:
        from grizli.aws import db as grizli_db
        dbFLAGS = grizli_db.FLAGS
    except:
        pass

    utils.set_warnings()

    #event = {'s3_object_path':'HST/Pipeline/j001452+091221/Extractions/j001452+091221_00277.beams.fits'}

    silent = False
    if 'silent' in event:
        silent = event['silent'] in TRUE_OPTIONS

    ###
    # Parse event arguments
    ###
    event_kwargs = {}
    for k in event:

        # Lists
        if isinstance(event[k], str):
            # Split lists
            if ',' in event[k]:
                try:
                    event_kwargs[k] = np.cast[float](event[k].split(','))
                except:
                    event_kwargs[k] = event[k].split(',')
            else:
                event_kwargs[k] = event[k]
        else:
            try:
                event_kwargs[k] = json.loads(event[k])
            except:
                event_kwargs[k] = event[k]

    # Defaults
    if 'skip_started' not in event_kwargs:
        event_kwargs['skip_started'] = True

    for k in ['quasar_fit', 'extract_from_flt', 'fit_stars', 'beam_info_only']:
        if k not in event_kwargs:
            event_kwargs[k] = False

    if event_kwargs['beam_info_only'] in TRUE_OPTIONS:
        dbtable = 'multibeam_v2'
    elif event_kwargs['quasar_fit'] in TRUE_OPTIONS:
        dbtable = 'redshift_fit_v2_quasar'
    elif event_kwargs['fit_stars'] in TRUE_OPTIONS:
        dbtable = 'stellar_fit_v2'
    else:
        dbtable = 'redshift_fit_v2'
    
    if 'dbtable' in event_kwargs:
        dbtable = event_kwargs['dbtable']
        
    if not silent:
        print('Grizli version: ', grizli.__version__)

    # Disk space
    total, used, free = shutil.disk_usage("/")
    if not silent:
        print('Disk info: Total = {0:.2f} / Used = {1:.2f} / Free = {2:.2f}'.format(total // (2**20), used // (2**20), free // (2**20)))

    # Output path
    if 'output_path' in event:
        output_path = event['output_path']
    else:
        output_path = None

    if 'bucket' in event:
        event_kwargs['bucket'] = event['bucket']
    else:
        event_kwargs['bucket'] = 'grizli-v2' #'aws-grivam'

    if 'working_directory' in event:
        os.chdir(event['working_directory'])
    else:
        os.chdir('/tmp/')

    if not silent:
        print('Working directory: {0}'.format(os.getcwd()))

    files = glob.glob('*')
    files.sort()

    # Filenames, etc.
    beams_file = os.path.basename(event['s3_object_path'])
    root = '_'.join(beams_file.split('_')[:-1])
    id = int(beams_file.split('_')[-1].split('.')[0])

    try:
        db_status = grizli_db.get_redshift_fit_status(root, id, table=dbtable)
    except:
        db_status = -1

    # Initial log
    start_log = '{0}_{1:05d}.start.log'.format(root, id)
    full_start = 'HST/Pipeline/{0}/Extractions/{1}'.format(root, start_log)
    if ((start_log in files) | (db_status >= 0)) & event_kwargs['skip_started']:
        print('Log file {0} found in {1} (db_status={2})'.format(start_log, os.getcwd(), db_status))
        return True

    if not silent:
        for i, file in enumerate(files):
            print('Initial file ({0}): {1}'.format(i+1, file))

    if os.path.exists('{0}/matplotlibrc'.format(grizli.GRIZLI_PATH)):
        os.system('cp {0}/matplotlibrc .'.format(grizli.GRIZLI_PATH))

    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bkt = s3.Bucket(event_kwargs['bucket'])
    
    ubkt = bkt
    if 'up_bucket' in event_kwargs:
        if event_kwargs['up_bucket'] != event_kwargs['bucket']:
            print('Set upload bucket: ' + event_kwargs['up_bucket'])
            ubkt = s3.Bucket(event_kwargs['up_bucket'])
            
    if event_kwargs['skip_started']:
        res = [r.key for r in ubkt.objects.filter(Prefix=full_start)]
        if res:
            print('Already started ({0}), aborting.'.format(start_log))
            return True

    fp = open(start_log, 'w')
    fp.write(time.ctime()+'\n')
    fp.close()
    ubkt.upload_file(start_log, full_start)

    # Download fit arguments
    if 'force_args' in event:
        force_args = event['force_args'] in TRUE_OPTIONS
    else:
        force_args = False

    args_files = ['{0}_fit_args.npy'.format(root), 'fit_args.npy']
    for args_file in args_files:
        if (not os.path.exists(args_file)) | force_args:
            aws_file = 'HST/Pipeline/{0}/Extractions/{1}'.format(root, args_file)
            try:
                bkt.download_file(aws_file, './{0}'.format(args_file),
                              ExtraArgs={"RequestPayer": "requester"})
                print('Use args_file = {0}'.format(args_file))
                break
            except:
                continue

    # If no beams file in the bucket, try to generate it
    put_beams = False
    try:
        if not os.path.exists(beams_file):
            bkt.download_file(event['s3_object_path'], './{0}'.format(beams_file), ExtraArgs={"RequestPayer": "requester"})
            put_beams = False
    except:
        print('Extract from GrismFLT object!')
        if 'clean' in event:
            if isinstance(event['clean'], str):
                run_clean = event['clean'].lower() in ['true', 'y', '1']
            else:
                run_clean = event['clean']
        else:
            run_clean = True

        try:
            # Extracting beams
            grizli_db.update_redshift_fit_status(root, id,
                                                status=dbFLAGS['start_beams'],
                                                table=dbtable)
        except:
            print('Set DB flag failed: start_beams')
            pass

        status = extract_beams_from_flt(root, event_kwargs['bucket'], id,
                                        clean=run_clean, silent=silent)

        # Garbage collector
        gc.collect()

        if status is False:
            return False
        else:
            beams_file = status[0]

        try:
            # Beams are done
            grizli_db.update_redshift_fit_status(root, id,
                                                 status=dbFLAGS['done_beams'],
                                                 table=dbtable)
        except:
            pass

        put_beams = True

        # upload it now
        output_path = 'HST/Pipeline/{0}/Extractions'.format(root)
        for outfile in status:
            aws_file = '{0}/{1}'.format(output_path, outfile)
            print(aws_file)
            bkt.upload_file(outfile, aws_file, 
                            ExtraArgs={'ACL': 'public-read'})

    if ('run_fit' in event) & (dbtable == 'redshift_fit_v2'):
        if event['run_fit'] in FALSE_OPTIONS:
            res = ubkt.delete_objects(Delete={'Objects': 
                                               [{'Key': full_start}]})

            try:
                grizli_db.update_redshift_fit_status(root, id,
                                                 status=dbFLAGS['no_run_fit'],
                                                 table=dbtable)
            except:
                pass

            return True

    utils.fetch_acs_wcs_files(beams_file, bucket_name=event_kwargs['bucket'])

    # Update the multibeam/beam_geometry tables
    if os.path.exists(beams_file):
        args = np.load(args_file, allow_pickle=True)[0]
        for arg in event_kwargs:
            if arg in args:
                args[arg] = event_kwargs[arg]

        grizli_db.multibeam_to_database(beams_file, Rspline=15, force=False,
                                        **args)

    if dbtable == 'multibeam_v2':
        # Done
        res = ubkt.delete_objects(Delete={'Objects': [{'Key': full_start}]})
        return True

    # Download WCS files
    # if event_kwargs['check_wcs']:
    #     # WCS files for ACS
    #     files = [obj.key for obj in bkt.objects.filter(Prefix='Pipeline/{0}/Extractions/j'.format(root))]
    #     for file in files:
    #         if 'wcs.fits' in file:
    #             if os.path.exists(os.path.basename(file)):
    #                 continue
    #
    #             bkt.download_file(file, os.path.basename(file),
    #                               ExtraArgs={"RequestPayer": "requester"})

    # Is zr in the event dict?
    # if 'zr' in event:
    #     zr = list(np.cast[float](event['zr']))
    # else:
    #     try:
    #         zr = np.load('fit_args.npy')[0]['zr']
    #     except:
    #         zr = np.load('fit_args.npy', allow_pickle=True)[0]['zr']

    # Directory listing
    files = glob.glob('*')
    files.sort()

    if not silent:
        for i, file in enumerate(files):
            print('File ({0}): {1}'.format(i+1, file))

    try:
        files = glob.glob('{0}_{1:05d}*R30.fits'.format(root, id))
        if (len(files) > 0) & (dbtable == 'redshift_fit_v2'):
            grizli_db.send_1D_to_database(files=files)
    except:
        print('Failed to send R30 to spec1d database')
        pass

    ###
    # Run the fit
    try:
        grizli_db.update_redshift_fit_status(root, id, table=dbtable,
                                        status=dbFLAGS['start_redshift_fit'])
    except:
        print('Set DB flag failed: start_redshift_fit')
        pass
    
    # Extra args from numpy file on s3
    if 's3_argsfile' in event_kwargs:
        if os.path.exists(event_kwargs['s3_argsfile']):
            result = (event_kwargs['s3_argsfile'], 1)
        else:
            result = utils.fetch_s3_url(url=event_kwargs['s3_argsfile'],
                                    file_func=lambda x : os.path.join('./',x), 
                                    skip_existing=True, verbose=True)
        
        s3_argsfile_local, status = result
        if os.path.exists(s3_argsfile_local):
            try:
                extra_args = np.load(s3_argsfile_local, allow_pickle=True)[0]
                
                for k in extra_args:
                    print(f'{s3_argsfile_local} {k}')
                    event_kwargs[k] = extra_args[k]
                
            except OSError:
                print(f'Failed to get keywords from {s3_argsfile_local}')
                
    if event_kwargs['quasar_fit'] in TRUE_OPTIONS:

        # Don't recopy beams file
        put_beams = False

        # Don't make line maps
        if 'min_line_sn' not in event_kwargs:
            event_kwargs['min_line_sn'] = np.inf

        # Don't make drizzled psfs
        if 'get_ir_psfs' not in event_kwargs:
            event_kwargs['get_ir_psfs'] = False

        # Fit line widths
        if 'get_line_width' not in event_kwargs:
            event_kwargs['get_line_width'] = True

        # sys_err
        if 'sys_err' not in event_kwargs:
            event_kwargs['sys_err'] = 0.05

        # Don't use photometry
        event_kwargs['phot_obj'] = None
        event_kwargs['use_phot_obj'] = False

        event_kwargs['fit_only_beams'] = True
        event_kwargs['fit_beams'] = False

        templ_args = {'uv_line_complex': True,
                      'broad_fwhm': 2800,
                      'narrow_fwhm': 1000,
                      'fixed_narrow_lines': True,
                      'Rspline': 15,
                      'include_reddened_balmer_lines': False}

        for k in templ_args:
            if k in event_kwargs:
                templ_args[k] = event_kwargs.pop(k)

        if templ_args['broad_fwhm'] < 0:
            use_simple_templates = True
            templ_args['broad_fwhm'] *= -1
        else:
            use_simple_templates = False

        print('load_quasar_templates(**{0})'.format(templ_args))
        q0, q1 = utils.load_quasar_templates(**templ_args)

        if use_simple_templates:
            x0 = utils.load_templates(full_line_list=['highO32'], continuum_list=['quasar_lines.txt', 'red_blue_continuum_noLya.txt'], line_complexes=False, fwhm=1000)

            x0 = utils.load_templates(full_line_list=[], continuum_list=['quasar_lines.txt', 'red_blue_continuum_noLya.txt', 'fsps_starburst_lines.txt'], line_complexes=False, fwhm=1000)

            # for t in q0:
            #     if 'bspl' in t:
            #         x0[t] = q0[t]

            # Rename templates
            x0['continuum spline'] = x0.pop('red_blue_continuum_noLya.txt')
            x0['quasar'] = x0.pop('quasar_lines.txt')
            x0['fsps_starburst'] = x0.pop('fsps_starburst_lines.txt')

            q0 = x0
            q1['continuum spline'] = x0['continuum spline']

        # Quasar templates with fixed line ratios
        # q0, q1 = utils.load_quasar_templates(uv_line_complex=True,
        #                                     broad_fwhm=2800, narrow_fwhm=1000,
        #                                     fixed_narrow_lines=True,
        #                                     Rspline=15)

        if 'zr' not in event_kwargs:
            event_kwargs['zr'] = [0.03, 6.8]
        if 'fitter' not in event_kwargs:
            event_kwargs['fitter'] = ['lstsq', 'lstsq']

        print('run_all_parallel: {0}'.format(event_kwargs))

        fitting.run_all_parallel(id, t0=q0, t1=q1, args_file=args_file,
                                 **event_kwargs)

        if output_path is None:
            #output_path = 'Pipeline/QuasarFit'.format(root)
            output_path = 'HST/Pipeline/{0}/Extractions'.format(root)

    elif event_kwargs['fit_stars'] in TRUE_OPTIONS:

        args = np.load(args_file, allow_pickle=True)[0]

        if 'psf' in event_kwargs:
            args['psf'] = event_kwargs['psf'] in TRUE_OPTIONS

        for k in ['fcontam', 'min_sens', 'sys_err']:
            if k in event_kwargs:
                print('Set arg {0}={1}'.format(k, event_kwargs[k]))
                args[k] = event_kwargs[k]

        # Load MultiBeam
        mb = multifit.MultiBeam(beams_file, **args)

        if 'fit_trace_shift' in args:
            if args['fit_trace_shift']:
                tr = mb.fit_trace_shift()

        if 'spline_correction' in event_kwargs:
            spline_correction = event_kwargs['spline_correction'] in TRUE_OPTIONS
        else:
            spline_correction = True

        if 'fit_background' in event_kwargs:
            fit_background = event_kwargs['fit_background'] in TRUE_OPTIONS
        else:
            fit_background = True

        if 'fitter' in event_kwargs:
            fitter = event_kwargs['fitter']
        else:
            fitter = 'lstsq'

        if 'Rspline' in event_kwargs:
            Rspline = event_kwargs['Rspline']
        else:
            Rspline = 15

        if Rspline == 15:
            logg_list = [4.5]
        else:
            logg_list = utils.PHOENIX_LOGG

        if 'add_carbon_star' in event_kwargs:
            add_carbon_star = event_kwargs['add_carbon_star']
        else:
            add_carbon_star = 25

        if 'use_phoenix' in event_kwargs:
            p = event_kwargs.pop('use_phoenix')
            if p in TRUE_OPTIONS:
                tstar = utils.load_phoenix_stars(logg_list=logg_list,
                                             add_carbon_star=add_carbon_star)
            else:
                tstar = utils.load_templates(stars=True,
                                             add_carbon_star=add_carbon_star)
        else:
            tstar = utils.load_phoenix_stars(logg_list=logg_list,
                                             add_carbon_star=add_carbon_star)

        kws = {'spline_correction': spline_correction,
               'fit_background': fit_background,
               'fitter': fitter,
               'spline_args': {'Rspline': Rspline}}

        print('kwargs: {0}'.format(kws))

        # Fit the stellar templates
        _res = mb.xfit_star(tstar=tstar, oned_args={}, **kws)

        _res[0].savefig('{0}_{1:05d}.star.png'.format(root, id))

        # Save log info
        fp = open('{0}_{1:05d}.star.log'.format(root, id), 'w')
        fp.write(_res[1])
        fp.close()

        if output_path is None:
            # output_path = 'Pipeline/QuasarFit'.format(root)
            output_path = 'HST/Pipeline/{0}/Extractions'.format(root)

    else:
        # Normal galaxy redshift fit
        res = fitting.run_all_parallel(id, get_output_data=True,
                                       fit_only_beams=True, fit_beams=False,
                                       args_file=args_file,
                                       **event_kwargs)
        
        if root == 'fresco-gds-med':
            try:
                mb, tfit = res[0], res[3]
                # 1D for FRESCO
                fig1 = mb.oned_figure(bin=2, tfit=tfit,
                                      show_rest=False, minor=0.1,
                                      figsize=(9,3), median_filter_kwargs=None,
                                      ylim_percentile=0.1)
                
                hdu, fig2 = mb.drizzle_grisms_and_PAs(fcontam=0.0, 
                                                     flambda=False,
                                                     kernel='point', size=16,
                                                     tfit=tfit, diff=False, 
                                                      mask_segmentation=False,
                                    fig_args=dict(mask_segmentation=False, 
                                                  average_only=False,
                                                  scale_size=1, 
                                                  cmap='viridis_r', 
                                                  width_ratio=0.11),
                                                    )
            
                fig1.savefig(f'{root}_{id:05d}.1d.png')
                fig2.savefig(f'{root}_{id:05d}.2d.png')
            except:
                print('FRESCO outputs failed')
                
        if output_path is None:
            output_path = 'HST/Pipeline/{0}/Extractions'.format(root)

    # Output files
    files = glob.glob('{0}_{1:05d}*'.format(root, id))
    
    for file in files:
        if ('beams.fits' not in file) | put_beams:
            aws_file = '{0}/{1}'.format(output_path, file)

            if event_kwargs['quasar_fit'] in TRUE_OPTIONS:
                # Don't copy stack
                if 'stack' in file:
                    continue

                # Add qso extension on outputs
                aws_file = aws_file.replace('_{0:05d}.'.format(id),
                                            '_{0:05d}.qso.'.format(id))

            print('Upload {0} -> {1}'.format(file, aws_file))

            bkt.upload_file(file, aws_file, ExtraArgs={'ACL': 'public-read'})

    # Put data in the redshift_fit database table
    try:
        if dbtable == 'stellar_fit':
            rowfile = '{0}_{1:05d}.star.log'.format(root, id)
        else:
            rowfile = '{0}_{1:05d}.row.fits'.format(root, id)

        if os.path.exists(rowfile):
            grizli_db.add_redshift_fit_row(rowfile, table=dbtable,
                                           verbose=True)

        # Add 1D spectra
        files = glob.glob('{0}_{1:05d}*1D.fits'.format(root, id))
        if (len(files) > 0) & (dbtable == 'redshift_fit_v2'):
            grizli_db.send_1D_to_database(files=files)

    except:
        print('Update row failed')
        pass

    # Remove start log now that done
    res = ubkt.delete_objects(Delete={'Objects': [{'Key': full_start}]})

    # Garbage collector
    gc.collect()


def clean(root='', verbose=True):
    import glob
    import os
    import matplotlib.pyplot as plt
    import gc

    plt.close('all')

    files = glob.glob(root+'*')
    files += glob.glob('*wcs.fits')
    files.sort()

    for file in files:
        print('Cleanup: {0}'.format(file))
        os.remove(file)

    gc.collect()


TESTER = 'HST/Pipeline/j001452+091221/Extractions/j001452+091221_00277.beams.fits'


def run_test(s3_object_path=TESTER):
    import os

    from importlib import reload
    import grizli.aws.lambda_handler
    reload(grizli.aws.lambda_handler)

    from grizli.aws.lambda_handler import run_grizli_fit, clean

    # event = {'s3_object_path': s3_object_path, 'verbose':'True'}

    obj = 'j224916m4432_02983'

    bucket = 'grizli'
    #root, id = obj.split('_')

    root = '_'.join(obj.split('_')[:-1])
    id = int(obj.split('_')[-1].split('.')[0])
    
    s3_object_path = 'HST/Pipeline/{0}/Extractions/{0}_{1:05d}.beams.fits'.format(root, int(id))
    event = {'s3_object_path': s3_object_path,
             'bucket': bucket,
             'verbose': 'True',
             'check_wcs': 'True',
             'zr': '0.1,0.3',
             }

    run_grizli_fit(event)

    beams_file = os.path.basename(event['s3_object_path'])
    #root = beams_file.split('_')[0]
    clean(root=root, verbose=True)


def handler(event, context):

    if 'handler_type' not in event:
        if 'scale_ab' in event:
            handler_type = 'drizzle_handler'
        else:
            handler_type = 'redshift_handler'
    else:
        handler_type = event.pop('handler_type')

    print('Handler type: ', handler_type)

    # Send the event to the appropriate function
    if handler_type == 'redshift_handler':
        redshift_handler(event, context)

    elif handler_type == 'show_version_handler':
        show_version(event, context)

    elif (handler_type == 'drizzle_handler'):
        from grizli.aws import aws_drizzler
        aws_drizzler.handler(event, context)


def redshift_handler(event, context):
    """
    Function for handling `run_grizli_fit` events
    """
    import matplotlib.pyplot as plt
    import traceback
    import time

    t0 = time.time()

    print(event)  # ['s3_object_path'], event['verbose'])
    # print(context)

    try:
        run_grizli_fit(event)
    except:
        print(traceback.format_exc(limit=2))
        pass

    # Clean up
    beams_file = os.path.basename(event['s3_object_path'])
    root = '_'.join(beams_file.split('_')[:-1])

    if 'clean' in event:
        run_clean = event['clean'] in TRUE_OPTIONS
    else:
        run_clean = True

    if run_clean:
        clean(root=root, verbose=True)

    t1 = time.time()

    return (event['s3_object_path'], t0, t1)


def show_version(event, context):
    import glob
    import grizli
    import eazy

    print('Event: ', event)

    print('grizli version: ', grizli.__version__)
    print('eazy version: ', eazy.__version__)

    import matplotlib
    print('matplotlibrc: ', matplotlib.matplotlib_fname())
    
    if 'files' in event:
        for file in event['files']:
            print('\n\n === File {0} === \n\n'.format(file))
            with open(file) as fp:
                lines = fp.readlines()
            
            print(''.join(lines))
    
    if 'glob' in event:
        _res = glob.glob(event['glob'])
        print('\n\n === glob {0} === \n\n'.format(event['glob']))
        for file in _res:
            print(file)
            
if __name__ == "__main__":
    handler('', '')
