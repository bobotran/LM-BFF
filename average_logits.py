import argparse
import os
import numpy as np

def get_logits(logits_dir, sorted_templates=None, n_models=None):
    '''Returns the logits in logits_dir as a list of numpy arrays'''
    templates_to_load = set()
    if sorted_templates is not None and n_models is not None:
        with open(sorted_templates) as fp:
            for _ in range(n_models):
                line = fp.readline()
                templates_to_load.add(line.split()[0])
                
    logits_arrs = []
    logit_filenames = os.listdir(logits_dir)
    for fn in logit_filenames:
        try:
            template_id = fn.split('-')[1]
        except IndexError:
            # Skip files in this directory that don't have specific structure
            continue
        if template_id in templates_to_load:
            print("Ensembling template {}".format(fn))
            fp = os.path.join(logits_dir, fn)
            logits_arrs.append(np.load(fp))
    return logits_arrs
    
def average_logits(logits_arrs):
    logits_arr = np.stack(logits_arrs, axis=0)
    return np.mean(logits_arr, axis=0, keepdims=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logits_dir', required=True, help="Logits in this folder will be averaged together into a new .npy file")
    parser.add_argument('--sorted_templates', default=None, help="If specified, picks the top n templates to ensemble.")
    parser.add_argument('--n_models', default=None, type=int, help="If specified, picks the top n templates to ensemble.")
    args = parser.parse_args()
    
    logits_arrs = get_logits(args.logits_dir, args.sorted_templates, args.n_models)
    avg_logits = average_logits(logits_arrs)
    
    np.save(os.path.join(args.logits_dir, 'avg_logits.npy'), avg_logits)