"""
Automated processing of associated exposures
"""

def fetch_from_AWS_bucket(root='j022644-044142', id=1161, product='.beams.fits', bucket_name='aws-grivam', verbose=True, dryrun=False, output_path='./', get_fit_args=False, skip_existing=True):
    """
    Fetch products from the Grizli AWS bucket.  
    
    Boto3 will require that you have set up your AWS credentials in, e.g., 
    ~/.aws/credentials
    """
    import os
    import boto3
    
    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bkt = s3.Bucket(bucket_name)
    
    files = [obj.key for obj in bkt.objects.filter(Prefix='Pipeline/{0}/Extractions/{0}_{1:05d}{2}'.format(root, id, product))]
    
    if get_fit_args:
        files += ['Pipeline/{0}/Extractions/fit_args.npy'.format(root)]
        
    for file in files:
        local = os.path.join(output_path, os.path.basename(file))
        
        if verbose:
            print('{0} -> {1}'.format(file, output_path))
            
        if not dryrun:
            if os.path.exists(local) & skip_existing:
                continue
            
            bkt.download_file(file, local, 
                          ExtraArgs={"RequestPayer": "requester"})
        
