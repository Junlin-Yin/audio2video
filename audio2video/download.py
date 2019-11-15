from subprocess import call
from __init__ import raw_dir

dir = '%s/mp4' % raw_dir
input_file = 'obama_addresses.txt'
command = "you-get --itag=136 -I " + input_file + " -o " + dir + " --no-caption -d "
call(command)