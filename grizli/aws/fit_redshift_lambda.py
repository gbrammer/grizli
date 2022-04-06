#!/usr/bin/env python
def fit_lambda(root='j100025+021706', beams=[], ids=[], newfunc=False, bucket_name='aws-grivam', skip_existing=True, sleep=True, skip_started=True, quasar_fit=False, output_path=None, show_event=False, **kwargs):
    """

    """
    import time
    import os
    import yaml

    import numpy as np
    import boto3
    import json

    if (len(beams) == 0) & (len(ids) == 0):
        beams, files = get_needed_paths(root, bucket_name=bucket_name, skip_existing=skip_existing)
    elif len(ids) > 0:
        # ids key takes precedence over beams
        beams = ['HST/Pipeline/{0}/Extractions/{0}_{1:05d}.beams.fits'.format(root, int(id)) for id in ids]
        if sleep == 1:
            sleep = False
    else:
        if sleep == 1:
            sleep = False

    if len(beams) == 0:
        print('{0}: No beams to fit'.format(root))

        return False
    # Auth to create a Lambda function (credentials are picked up from above .aws/credentials)
    session = boto3.Session()

    # Make sure Lambda is running in the same region as the HST public dataset
    client = session.client('lambda', region_name='us-east-1')

    # Lambda Function names
    func = 'GrizliTestFunction'
    func = 'GrizliLambda-0-12-0-41'
    func = 'GrizliLambda2020'

    print('Lambda function: {0} (newfunc={1})'.format(func, newfunc))

    # Auth to create a Lambda function
    session = boto3.Session()
    client = session.client('lambda', region_name='us-east-1')

    # s3 object = s3://{bucket_name}/{s3_object_path}
    # e.g., 'Pipeline/sparcs0034/Extractions/sparcs0034_00441.beams.fits'
    all_events = []

    NB = len(beams)
    for i, s3_object_path in enumerate(beams):
        print('{0:>5} / {1:>5} : {2}'.format(i+1, NB, s3_object_path))

        if False:
            # Defaults
            skip_started = True  # SKip objects already started
            quasar_fit = False  # Fit with quasar templates

        event = {
              "s3_object_path": s3_object_path,
              "bucket": bucket_name,
              "skip_started": skip_started,
              "quasar_fit": quasar_fit,
            }

        for arg in kwargs:
            event[arg] = kwargs[arg]  # str(arg)

        if output_path is not None:
            if output_path == 'self':
                event['output_path'] = os.path.dirname(s3_object_path)
            else:
                event['output_path'] = output_path

        for k in event:
            if isinstance(event[k], (list, np.ndarray)):
                event[k] = ','.join(['{0}'.format(a) for a in event[k]])
            else:
                pass

                # try:
                #     event[k] = json.dumps(event[k])
                # except:
                #     print('Couldn\'t json.dumps item: ', event[k])

            # if isinstance(event[k], bool):
            #     event[k] = str(event[k])
            #
            # if isinstance(event[k], dict):
            #     event[k] = json.dumps(event[k])

        if show_event == 1:
            print('Event: \n\n', event.__str__().replace('\'', '"').replace(',', ',\n'))
            response = client.invoke(FunctionName=func,
                                     InvocationType='Event', LogType='Tail',
                                     Payload=json.dumps(event))
        if show_event == 2:
            all_events.append(event)
        else:
            # Invoke Lambda function
            response = client.invoke(FunctionName=func,
                                     InvocationType='Event', LogType='Tail',
                                     Payload=json.dumps(event))

    if show_event == 2:
        return all_events

    # Sleep for status check
    beams, full, logs, start = get_needed_paths(root, bucket_name=bucket_name, skip_existing=True, get_lists=True)
    if (sleep) & (len(beams) > len(full)) & (len(beams) > 0):
        sleep_time = 303*np.ceil(len(beams)/950)
        print('{0}: sleep {1}'.format(time.ctime(), sleep_time))

        time.sleep(sleep_time)

        # Status again to check products

        # Wait up to an extra 10 minutes checking if beams finish
        iter = 0
        while (len(beams) > len(full)) & (len(start) > 0) & (iter < 10):
            iter += 1
            time.sleep(61)
            beams, full, logs, start = get_needed_paths(root, bucket_name=bucket_name, skip_existing=True, get_lists=True)


BASE_EVENT = {'bucket': 'grizli-v2', 'skip_started': True, 'quasar_fit': False, 'zr': '0.01,3.2', 'force_args': True}


def generate_events(roots, ids, base=BASE_EVENT, send_to_lambda=False):
    """
    Generate many events with `s3_object_path` based on roots/ids
    """
    events = []
    for root, id in zip(roots, ids):
        event = base.copy()
        event['s3_object_path'] = 'HST/Pipeline/{0}/Extractions/{0}_{1:05d}.beams.fits'.format(root, id)
        events.append(event)

    if send_to_lambda:
        client = get_lambda_client()
        for event in events:
            send_event_lambda(event, verbose=True, client=client, func='GrizliLambda2020')

    return events


def get_lambda_client(region_name='us-east-1'):
    """
    Get boto3 client in same region as HST public dataset
    """
    import boto3
    session = boto3.Session()
    client = session.client('lambda', region_name=region_name)
    return client


def send_event_lambda(event, verbose=True, client=None, func='grizli-redshift-fit'):
    """
    Send a single event to AWS lambda
    
    GrizliLambda2020
    """
    import time
    import os
    import yaml

    import numpy as np
    import boto3
    import json

    if client is None:
        client = get_lambda_client(region_name='us-east-1')

    if ('output_path' in event) & ('s3_object_path' in event):
        if event['output_path'] == 'self':
            event['output_path'] = os.path.dirname(event['s3_object_path'])

    if verbose:
        print('Send event to {0}: {1}'.format(func, event))

    response = client.invoke(FunctionName=func,
                             InvocationType='Event', LogType='Tail',
                             Payload=json.dumps(event))


def get_needed_paths(root, get_string=False, bucket_name='aws-grivam', skip_existing=True, get_lists=False):
    """
    Get the S3 paths of the "beams.fits" files that still need to be fit.
    """
    import boto3
    import time

    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bkt = s3.Bucket(bucket_name)

    files = [obj.key for obj in bkt.objects.filter(Prefix='HST/Pipeline/{0}/Extractions/'.format(root))]

    beams = []
    logs = []
    full = []
    start = []

    for file in files:
        if 'beams.fits' in file:
            beams.append(file)

        if 'log_par' in file:
            logs.append(file)

        if 'start.log' in file:
            start.append(file)

        if 'full.fits' in file:
            full.append(file)

    label = '{0} / {1} / Nbeams: {2}, Nfull: {3}, Nlog: {4}, Nstart: {5}'.format(root, time.ctime(), len(beams), len(full), len(logs), len(start))
    if get_string:
        return label

    print(label)

    if get_lists:
        return beams, full, logs, start

    for i in range(len(beams))[::-1]:
        test = (beams[i].replace('.beams.fits', '.full.fits') in full)
        test |= (beams[i].replace('.beams.fits', '.start.log') in start)
        if test & skip_existing:
            beams.pop(i)

    return beams, files


if __name__ == "__main__":
    import sys
    import yaml

    import numpy as np
    from grizli import utils
    from grizli.pipeline import auto_script
    utils.set_warnings()

    if len(sys.argv) < 2:
        print('Usage: fit_redshift_lambda.py {field}')
        exit

    root = sys.argv[1]

    # bucket_name = 'grizli-grism'
    bucket_name = 'aws-grivam'
    skip_existing = True
    newfunc = False

    kwargs = {'root': root,
              'bucket_name': bucket_name,
              'skip_existing': True,
              'newfunc': False,
              'sleep': True,
              'beams': [],
              'output_path': None,
              'zr': None,
              'show_event': False}

    # Args passed to the lambda event
    kwargs['skip_started'] = True  # SKip objects already started
    kwargs['quasar_fit'] = False  # Fit with quasar templates

    dryrun = False

    if len(sys.argv) > 2:
        for args in sys.argv[2:]:
            keypair = args.strip('--').split('=')

            # Booleans
            if keypair[0] in ['newfunc', 'skip_existing', 'sleep', 'skip_started', 'quasar_fit', 'show_event']:
                if len(keypair) == 1:
                    kwargs[keypair[0]] = True
                else:
                    kwargs[keypair[0]] = keypair[1].lower() in ['true']

            # List of s3 beams.fits paths
            elif keypair[0] == 'beams':
                # Fit a single object
                kwargs['beams'] = keypair[1].split(',')

            # List of ids associated with {root}
            elif keypair[0] == 'ids':
                kwargs['beams'] = ['HST/Pipeline/{0}/Extractions/{0}_{1:05d}.beams.fits'.format(root, int(id)) for id in keypair[1].split(',')]

            # don't run the script
            elif keypair[0] == 'dryrun':
                dryrun = True

            # Everything else
            else:
                if keypair[0] in kwargs:
                    kwargs[keypair[0]] = keypair[1]

    print('Arguments: \n\n', '  '+yaml.dump(kwargs).replace('\n', '\n   '))

    if not dryrun:
        fit_lambda(**kwargs)  # newfunc=newfunc, bucket_name=bucket_name, skip_existing=skip_existing)
