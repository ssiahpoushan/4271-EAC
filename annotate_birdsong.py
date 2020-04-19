import numpy as np
import librosa

def load_data(filename, read_path):
    '''Load a WAV file and split it into one second-long blocks.

    Inputs
    ------
    filename : string - name of the file
    read_path : string - directory containing the files

    Returns
    -------
    2d array, shape (n_samples, 48000) - the data matrix'''

    audio = librosa.load('{}/{}.WAV'.format(read_path, filename), 48000)[0]
    return audio.reshape(-1, 48000)

def extract_features(X):
    '''Perform feature extraction on raw audio examples.

    Features 0 to 25 are MFCC statistics, 26 & 27 are zero crossing rate
    statistics, and 28 & 29 are spectral centroid statistics.
    Even-indexed features are frame means, odd-indexed features are frame
    standard deviations.

    Inputs
    ------
    X : 2d array, shape (n_samples, 48000) The data matrix

    Returns
    -------
    2d array, shape (n_samples, 30) The feature matrix'''

    n_mfcc = 13
    X_mfcc = np.empty((X.shape[0], 2 * n_mfcc))
    for i, x in enumerate(X):
        mfcc = librosa.feature.mfcc(x, n_mfcc=n_mfcc)
        X_mfcc[i] = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])

    X_zcr = np.empty((X.shape[0], 2))
    for i, x in enumerate(X):
        zcr = librosa.feature.zero_crossing_rate(x)
        X_zcr[i] = [zcr.mean(), zcr.std()]

    X_sc = np.empty((X.shape[0], 2))
    for i, x in enumerate(X):
        sc = librosa.feature.spectral_centroid(x)
        X_sc[i] = [sc.mean(), sc.std()]

    return np.hstack([X_mfcc, X_zcr, X_sc])

def load_model():
    '''Load the necessary model parameters from file.

    Returns
    -------
    sklearn.preprocessing.StandardScaler - scaling object
    sklearn.svm.SupportVectorClassifier - classifying object'''

    from joblib import load
    return load('model/scaler.joblib'), load('model/svm.joblib')

def annotate(y, filename, write_path):
    '''Translate sample labels into timestamp annotations and write to a file.

    Inputs
    ------
    y : array, shape (n_samples) - predicted label for each one second sample
    filename : string - the name of the output file
    write_path : the target directory in which to place the annotation files'''
    annotations = []
    duration = 0
    for i in range(len(y)):
        if y[i]:
            if duration == 0:
                start = i
            duration += 1
        elif duration > 0:
            annotations += [[start * 1000, (start + duration) * 1000 - 1]]
            duration = 0
    if duration > 0:
        annotations += [[start * 1000, (start + duration) * 1000 - 1]]
    with open('{}/{}.txt'.format(write_path, filename), 'w') as file:
        for anno in annotations:
            file.write('{} {}\n'.format(anno[0], anno[1]))

if __name__ == '__main__':
    import sys

    read_path = ''
    write_path = ''
    args = sys.argv[1:]
    start = 0
    verbose = False
    for i, arg in enumerate(args):
        if arg == '-h':
            print('Usage: annotate_birdsong [-FLAGS] [WAV file 1] [WAV file 2] ...')
            print('File names do not need to include extensions, but should refer to \nWAV files in the current directory (or specified by read path)')
            print('Flags:')
            print('-rp - read path for audio files')
            print('-wp - write path for annotation files')
            print('-v  - verbose output')
            print('-h  - help (print this message)')
            args = []
            break
        elif arg == '-v':
            verbose = True
            start += 1
        elif arg == '-rp':
            read_path = args[i+1]
            start += 2
        elif arg == '-wp':
            write_path = args[i+1]
            start += 2

    filenames = args[start:]
    for filename in filenames:
        X = load_data(filename, read_path)
        if verbose:
            print('Audio file {}.WAV loaded!'.format(filename))

        scaler, svm = load_model()
        if verbose:
            print('Model loaded!')

        X_feat = extract_features(X)
        if verbose:
            print('Features extracted!')

        y = svm.predict(scaler.transform(X_feat))
        if verbose:
            print('Predictions made!')

        annotate(y, filename, write_path)
        if verbose:
            print('Annotations written to file.')
