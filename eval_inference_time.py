import signal
import os
import pickle
import json

from job import presets
from paths_config import paths_config

results_filename = os.path.join(paths_config["results_folder"], paths_config["eval_results_filename"])
timing_filename =  os.path.join(paths_config["temp_folder"], "timing.pickle")

if __name__ == "__main__":
    # Are there previous results?
    if os.path.exists(results_filename):
        # Yes, load from pickle
        with open(results_filename, "rb") as f:
            results = pickle.load(f)
        if "largest_inference_batch_sizes" not in results:
            results["largest_inference_batch_sizes"] = {}
        if "inference_step_times" not in results:
            results["inference_step_times"] = {}
    else:
        raise Exception(f"No results file ({results_filename}) to save results to. See README.md for the workflow to follow.")

    if "largest_inference_batch_sizes" not in results:
        results["largest_inference_batch_sizes"] = {}
    if "inference_step_times" not in results:
        results["inference_step_times"] = {}

    # Optionally reset results
    #results["largest_inference_batch_sizes"] = {}
    #results["inference_step_times"] = {}

    for preset in presets.keys():
        print(f"Preset: {preset}")

        if preset not in results["largest_inference_batch_sizes"]:
            current = 8192  # Initial guess
            step = current
            largest_inference_batch_size = 1
            doubling = True
            while True:
                print(f"Current: {current}, largest_inference_batch_size: {largest_inference_batch_size}")
                retval = os.system(f"python job.py preset={preset} folder= save=False batch_size={current} inference_steps=5")
                print(f"retval: {retval}")
                if retval == signal.SIGINT:
                    signal.raise_signal(retval)
                if retval == 0:
                    # Success
                    largest_inference_batch_size = current
                    if doubling:
                        current = current*2
                        step = current//2
                    else:
                        step //= 2
                        if step == 0:
                            break
                        current = current + step
                else:
                    # Failure
                    doubling = False
                    step //= 2
                    if step == 0:
                        break
                    current = current - step
                    if current == largest_inference_batch_size:
                        break

            print(f"Preset {preset}, largest_inference_batch_size: {largest_inference_batch_size}")
            results["largest_inference_batch_sizes"][preset] = largest_inference_batch_size

        if preset not in results["inference_step_times"]:
            while True:
                use_batch_size = results["largest_inference_batch_sizes"][preset]
                if os.path.exists(timing_filename):
                    os.remove(timing_filename)
                retval = os.system(f"python job.py preset={preset} folder= save=False timing_filename={timing_filename} batch_size={use_batch_size} inference_steps=1000 cooldown=30")
                if retval == signal.SIGINT:
                    if os.path.exists(timing_filename):
                        os.remove(timing_filename)
                    signal.raise_signal(retval)
                if retval == 0:
                    if os.path.exists(timing_filename):
                        with open(timing_filename, "rb") as f:
                            timing = pickle.load(f)
                        results["inference_step_times"][preset] = timing
                        os.remove(timing_filename)
                        break
                    else:
                        raise Exception(f"No timing result file ""{timing_filename}""found")
                results["largest_inference_batch_sizes"][preset] -= 1

        print("largest_inference_batch_size:")
        print(json.dumps(results["largest_inference_batch_sizes"][preset], indent=4))
        if preset in results["inference_step_times"]:
            print("inference_step_times:")
            print(json.dumps(results["inference_step_times"][preset], indent=4))

        with open(results_filename, "wb") as f:
            # Pickle result
            pickle.dump(results, f)

    print("largest_inference_batch_sizes:")
    print(json.dumps(results["largest_inference_batch_sizes"], indent=4))
    print("inference_step_times")
    print(json.dumps(results["inference_step_times"], indent=4))