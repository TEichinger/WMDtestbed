import pickle
import argparse
import sys

sys.path.append('../')
sys.path.append('../wmd/python-emd-master')

from util.utilities import load_pickle, extract_unique_txtfiles_from_dirs


# extract the signature words/weights for a given file (@ 'file_path')
def inspect_signature_words(file_path, tfidf_triple_name=None, sigvec_strat=None, sigweight_strat=None, sig_size=10,
                            min_df=0.0, max_df=1.0, verbose=True):
    # load the text
    with open(file_path) as f:
        document = f.read()

    # PREPROCESSING (such as drop HTML tags etc.)
    #############################################
    all_word_tokens, _ = custom_tokenize(document)
    word_tokens = list(set(all_word_tokens))

    # get occurrence counts for the words in word_tokens
    term_counts = [all_word_tokens.count(word) for word in word_tokens]

    # PICK SIGNATURE WORDS
    #######################
    sigvec_strat, word_tokens = apply_signature_vector_strategy(word_tokens, term_counts, sig_size,
                                                                tfidf_triple_name=tfidf_triple_name,
                                                                sigvec_strat=sigvec_strat, min_df=min_df, max_df=max_df)

    # ASSIGN WORD WEIGHTS
    #####################
    sigweight_strat, word_weights = apply_signature_weight_strategy(word_tokens, term_counts,
                                                                    tfidf_triple_name=tfidf_triple_name,
                                                                    sigweight_strat=sigweight_strat, min_df=min_df,
                                                                    max_df=max_df)

    sorted_weights_and_tokens = sorted(zip(word_weights, word_tokens), reverse=True)

    word_tokens = [el[1] for el in sorted_weights_and_tokens]
    word_weights = [el[0] for el in sorted_weights_and_tokens]

    # PRINT SIGNATURE WORDS AND WEIGHTS, IF VERBOSE
    ###############################################
    if verbose:
        print(word_tokens)
        print(word_weights)

    return word_tokens, word_weights


def inspect_dropped_words(file_path, verbose=False):
    """ Instead of inspecting the words to valvulate, inspect the words dropped by <custom_tokenize>, actually we would also have to
    consider the words dropped by min_df, and max_df when sig_vec_strat is 'tfidf', yet this is omitted for now."""
    # load the text
    with open(file_path) as f:
        document = f.read()

    # PREPROCESSING (such as drop HTML tags etc.)
    #############################################
    all_word_tokens, stop_words_used = custom_tokenize(document)

    # PRINT SIGNATURE WORDS AND WEIGHTS, IF VERBOSE
    ###############################################
    if verbose:
        print(stop_words_used)

    return stop_words_used


def main():
    # PARSE ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--filedirs",
                        help="semicolon-separated string of paths to directories with pickle files to 'pickle' and inspect.",
                        action="store")
    parser.add_argument("--filepath",
                        help="path to a text file; if sigweight_strat or sigvec_strat = tfidf, a tfidf_triple_name has to be specified",
                        action="store")
    parser.add_argument("--picklepath", help="path to a signature pickle file", action="store")
    parser.add_argument("--show_stop_words", help="'1':show stop_words instead of words used for the WMD",
                        action="store")

    args = parser.parse_args()

    filedirs = args.filedirs
    filepath = args.filepath
    picklepath = args.picklepath
    show_stop_words = args.show_stop_words

    if show_stop_words == '1':
        show_stop_words = True

    if filedirs:
        filedirs = filedirs.split(';')
        pickle_files_to_inspect = extract_unique_txtfiles_from_dirs(filedirs)

        for file_path in pickle_files_to_inspect:
            word_tokens, word_weights = inspect_signature_words(file_path, tfidf_triple_name="test",
                                                                sigvec_strat="yake", sigweight_strat="", sig_size=100,
                                                                min_df=0.0, max_df=1.0, verbose=False)
            # print("Only the top 10 words:")
            # print("Len of sig words", len(word_tokens))
            # print(word_tokens[:10])

            if show_stop_words:
                stop_words_used = inspect_dropped_words(file_path, verbose=False)
                print(stop_words_used)

    # if a filepath has been specified, use a generic parameter setting for signature creation, prompt it afterwards
    if filepath:
        word_tokens, word_weights = inspect_signature_words(filepath, tfidf_triple_name="test", sigvec_strat="yake",
                                                            sigweight_strat="", sig_size=100, min_df=0.0, max_df=1.0,
                                                            verbose=False)
        print("Only the top 10 words:")
        print(word_tokens[:10])
    # if a picklepath has been specified, load the pickle file and prompt it
    if picklepath:
        loaded_pickle = load_pickle(picklepath)
        print(loaded_pickle)


if __name__ == "__main__":
    main()
