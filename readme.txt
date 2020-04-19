Birdsong Annotator

This repository contains code which automatically detects and annotates likely
birdsong in an WAV audio file. Annotations are output as timestamps in a text
file with the same name as the input file(s), where annotations are given in the
following format:

[start time (ms)] [stop time(ms)]

For example:

11000 15999

This denotes a birdlike sound beginning at 11.41 seconds and terminating at
15.99 seconds.

The audio file is processing in 1-second long blocks, so sub-1-second resolution
is not currently supported.

Usage: python annotate_birdsong.py [FLAGS] [FILENAME 1] [FILENAME 2] [...]

where flags are any of the following:
-h : print a help message
-v : turn on verbose output
-rp : set the read path (location of audio files)
-wp : the write path (destination of annotation files)

For example:

python annotate_birdsong.py -v -rp data/audio -wp data/output 5E63D5EF 5E63D66B

will look for files named 5E63D5EF.WAV and 5E63D66B.WAV in the directory
data/audio and annotate birdlike sounds in files named 5E63D5EF.WAV and
5E63D66B.WAV in the directory data/output.

Dependencies: numpy, librosa, scikit-learn, joblib
