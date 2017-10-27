"""
# setup version
Optional script to be run after running grizli's setup.py.
Performs same functions as `Build grizli` section in
http://grizli.readthedocs.io/en/master/grizli/install.html

Assumes $HOME properly set in .bashrc or .bash_profile

Nominal Use:

    > cd grizli/grizli
    > python post_install.py 

Optional Use (if want script to create 'iref' or 'jref' directories):

    > python post_install.py --iref 'desired/path/to/iref'

Option Use (if want script to modify your .bashrc with paths to iref, jref, and GRIZLI):

    > python post_install --modify_bashrc True
"""

from __future__ import print_function
import argparse
import os
import shutil
import subprocess
import time

def post_install(path_GRIZLI, iref, jref, modify_bashrc):

    print("Begun running grizli's post-installation script.")
    print(" ")

    # Set path to GRIZLI if not provided by user.
    home = os.getenv("HOME")
    if path_GRIZLI == None:
        path_GRIZLI = os.path.join(home, 'GRIZLI')

    # Create the GRIZLI directory.
    if os.path.isdir(path_GRIZLI):
        print("ERROR: {} already exists".format(path_GRIZLI))
        print("Remove the directory if you want to re-generate it.")
        print("Also be sure to check $GRIZLI in your .bashrc points to ")
        print("the correct location.")
        return
    elif not os.path.isdir(path_GRIZLI):
        os.mkdir(path_GRIZLI)
        print("Created directory: {}".format(path_GRIZLI))

    # Create subdirectories of GRIZLI.
    for subdir in ['CONF', 'templates']:
        path_subdir = os.path.join(path_GRIZLI, subdir)
        if not os.path.isdir(path_subdir):
            os.mkdir(path_subdir)
            print("Created directory: {}".format(path_subdir))        

    # Create iref and jref directories if provided by the user.
    if iref not None:
        iref_path = os.path.join(iref, 'iref')
        if not os.path.isdir(iref_path):
            os.mkdir(iref_path)
            print("Created directory: {}".format(iref_path))
    if jref not None:
        jref_path = os.path.join(jref, 'jref')
        if not os.path.isdir(jref_path):
            os.mkdir(jref_path)
            print("Created directory: {}".format(jref_path))

    print(" ")

    if modify_bashrc:
        print("The directory 'GRIZLI' will be added your HOME, {}".format(home))
        print(" ")
        with open(os.path.join(home, ".bashrc"), "r") as f:
            lines = f.readlines()
        with open(os.path.join(home, "temp.bashrc"), "w") as f:
            for line in lines:
                f.write(line)
            f.write('# Exports created by grizli/post_install.py\n')
            f.write('export GRIZLI="' + path_GRIZLI +'"\n')
            if iref not None:
                f.write('export iref="' + iref_path + '"\n')
            if jref not None:
                f.write('export jref="' + jref_path + '"\n')
        shutil.copy(os.path.join(home, ".bashrc"), os.path.join(home, "original.bashrc"))
        shutil.move(os.path.join(home, "temp.bashrc"), os.path.join(home, ".bashrc"))

        print("Added GRIZLI to ~/.bashrc.")
        print("Copied your original ~/.bashrc to ~/original.bashrc.")
        print("Remember to do 'source ~/.bashrc'")
        print(" ")

    else:
        print("Add the following line(s) to your .bashrc: ")
        print("export GRIZLI='{}'".format(path_GRIZLI))
        if iref not None:
            print("export iref='{}'".format(iref_path))
        if jref not None:
            print("export jref='{}'".format(jref_path))

    import grizli
    if iref not None:
        grizli.utils.fetch_default_calibs(ACS=False, 
            iref_path=iref_path)
        print("Fetched iref files.")
    if jref not None:
        grizli.utils.fetch_default_calibs(ACS=True, 
            jref_path=jref_path)
        print("Fetched jref files.")
    grizli.utils.fetch_config_files()            # to $GRIZLI/CONF
    print("Fetched CONF files.")
    print(" ")

    for item in os.listdir('data/templates'):
        os.symlink(os.path.join(os.path.realpath('data/templates'), item), 
        os.path.join(path_GRIZLI, 'templates', item))   

    print("Created symbolic link from grizli/data/templates to {}"\
        .format(os.path.join(path_GRIZLI, 'templates')))
    print(" ")
    print("INSTALLATION COMPLETE")
    print(time.ctime())
    print(" ")
    print("You should have this file structure in your HOME: ")
    print("    GRIZLI/")
    print("           CONF/")
    print("           templates/")
    print("And you should have iref/ and jref/ directories somewhere.")
    print("If elected to modify the .bashrc you should see the lines: ")
    print("     export GRIZLI='{}'".format(path_GRIZLI))
    print("     export iref='{}'".format(path_iref))
    print("     export jref='{}'".format(path_jref))
    print(" ")
    print("You should feel free to rearrange directories and appropriately ")
    print("change paths in the .bashrc as you see fit.")


def parse_args():
    """Parses command line arguments.
        
    Returns
    -------
    args : object
        Containing the image and destination arguments.
            
    """

    grizli_path_help = "(Optional) Path to where you want GRIZLI configuration files to be located."
    grizli_path_help += "If you do not supply a path, the script will create a 'GRIZLI' directory in your HOME." 
    iref_help = "(Optional) Path you want 'iref' directory containing WFC3 reference files to be created."
    iref_help += "(This can be the same path as to GRIZLI config files.)"
    iref_help += "Skip if you already have an iref directory set."
    jref_help = "(Optional) Path you want 'jref' directory containing ACS reference files to be created."
    iref_help += "(This can be the same path as to GRIZLI config files.)"
    jref_help += "Skip if you already have an jref directory set."
    modify_bashrc_help = "(Optional) Set to 'True' if you want the script to modify your .bashrc with iref, jref, and GRIZLI paths."
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--grizli_path', dest = 'grizli_path',
                        action = 'store', type = str, required = False,
                        help = grizli_path_help, default = None)
    parser.add_argument('--iref', dest = 'iref', 
                        action = 'store', type = str, required = False,
                        help = iraf_help, default = None)
    parser.add_argument('--jref', dest = 'jref', 
                        action = 'store', type = str, required = False,
                        help = jraf_help, default = None)
    parser.add_argument('--modify_bashrc', dest = 'modify_bashrc', 
                        action = 'store', type = str, required = False,
                        help = modify_bashrc_help, default = False)

    args = parser.parse_args()
     
    return args


if __name__=='__main__':
    args = parse_args()
    grizli_path = args.grizli_path
    iref = args.iref
    jref = args.jref
    modify_bashrc = args.modify_bashrc

    post_install(grizli_path, iref, jref, modify_bashrc)
