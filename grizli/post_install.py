"""
Optional script to be run after running grizli's setup.py.
Performs same functions as `Build grizli` section in
http://grizli.readthedocs.io/en/master/grizli/install.html

Assumes $HOME properly set in .bashrc or .bash_profile

Use:

    > cd grizli/grizli
    > python post_install.py 

"""

from __future__ import print_function
import os
import shutil
import subprocess
import time

def post_install():

    home = os.getenv("HOME")
    print("Begun running grizli's post-installation script.")
    print(" ")
    print("The directory 'GRIZLI' will be added your HOME, {}".format(home))
    print(" ")
    path_GRIZLI = os.path.join(home, 'GRIZLI')
    if os.path.isdir(path_GRIZLI):
        print("ERROR: {} already exists".format(path_GRIZLI))
        print("Remove the directory if you want to re-generate it.")
        print("Also be sure to check $GRIZLI in your .bashrc points to ")
        print("the correct location.")
        return

    if not os.path.isdir(path_GRIZLI):
        os.mkdir(path_GRIZLI)
        print("Created directory: {}".format(path_GRIZLI))

    for subdir in ['CONF', 'templates', 'iref', 'jref']:
        path_subdir = os.path.join(path_GRIZLI, subdir)
        if not os.path.isdir(path_subdir):
            os.mkdir(path_subdir)
            print("Created directory: {}".format(path_subdir))        

    print(" ")

    with open(os.path.join(home, ".bashrc"), "r") as f:
        lines = f.readlines()
    with open(os.path.join(home, "temp.bashrc"), "w") as f:
        for line in lines:
            f.write(line)
        f.write('# Exports created by grizli/post_install.py\n')
        f.write('export GRIZLI="' + path_GRIZLI +'"\n')
        # is it worth having a check that iref, jref already exist?
        # user can always modify later...
        f.write('export iref="${GRIZLI}/iref/"\n')
        f.write('export jref="${GRIZLI}/jref/"\n')
    shutil.copy(os.path.join(home, ".bashrc"), os.path.join(home, "original.bashrc"))
    shutil.move(os.path.join(home, "temp.bashrc"), os.path.join(home, ".bashrc"))

    print("Added GRIZLI to ~/.bashrc.")
    print("Copied your original ~/.bashrc to ~/original.bashrc.")
    print(" ")

    import grizli
    grizli.utils.fetch_default_calibs(ACS=False, 
        iref_path=os.path.join(path_GRIZLI, 'iref'), 
        jref_path=os.path.join(path_GRIZLI, 'jref')) # to iref/iref
    print("Fetched iref files.")
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
    print("Remember to do 'source ~/.bashrc'")
    print(" ")
    print("You should have this file structure in your HOME: ")
    print("    GRIZLI/")
    print("           CONF/")
    print("           iref/")
    print("           jref/")
    print("           templates/")
    print("And you will have a modified .bashrc with lines: ")
    print("     export GRIZLI='{}'".format(path_GRIZLI))
    print("     export iref='${GRIZLI}/iref/'")
    print("     export jref='${GRIZLI}/jref/'")
    print(" ")
    print("You should feel free to rearrange directories and appropriately ")
    print("change paths in the .bashrc as you see fit.")

if __name__=='__main__':
    post_install()
