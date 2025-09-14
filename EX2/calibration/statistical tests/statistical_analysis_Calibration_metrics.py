import numpy as np
from scipy import stats
from scipy.stats import monte_carlo_test
import ast
import csv
import pandas as pd
import matplotlib.pyplot as plt

def write_statistics(all_statistics_t_test, all_statistics_wilcoxon, header, normal_p_values, dataset, model, is_brier_only=False):
    """ Method used to write the calculated statistics to CSV file for CodeBERT results.
    """
    filename=f"aggregated_statistics_{dataset}_{model}_brier_appended.csv"
    normal_p_values_uncal = []
    normal_p_values_cal = []

    for p_value_cal, p_value_uncal in normal_p_values:
        normal_p_values_cal.append(p_value_cal)
        normal_p_values_uncal.append(p_value_uncal)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["test"] + header)
        writer.writerow(["paired_t_test"] + all_statistics_t_test)
        writer.writerow(["wilcox_test"] + all_statistics_wilcoxon)
        writer.writerow(["p_values for normality_cal"] + normal_p_values_cal)
        writer.writerow(["p_values for normality_uncal"] + normal_p_values_uncal)

def write_LApredict_and_deepJIT(all_statistics_t_test_val, all_statistics_t_test_test, all_statistics_wilcoxon_val, all_statistics_wilcoxon_test, all_p_values_normality_val, all_p_values_normality_test, headers, dataset, model="LApredict"):
    """ Method used to write the calculated statistics to CSV file for LApredict and DeepJIT results.
    """
    # Validation FOlds
    filename=f"aggregated_statistics_validation_{dataset}_{model}.csv"
    new_headers = []
    p_values_cal_val = []
    p_values_uncal_val = []
    p_values_cal_test = []
    p_values_uncal_test = []
    for index, header in enumerate(headers):
        suffix_platt = "_platt"
        suffix_temp = "_temp"
        new_headers.append(header + suffix_platt)
        new_headers.append(header + suffix_temp)

    for p_value_cal, p_value_uncal in all_p_values_normality_val:
        p_values_cal_val.append(p_value_cal)
        p_values_uncal_val.append(p_value_uncal)

    for p_value_cal, p_value_uncal in all_p_values_normality_test:
        p_values_cal_test.append(p_value_cal)
        p_values_uncal_test.append(p_value_uncal)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["test"] + new_headers)
        writer.writerow(["paired_t_test"] + all_statistics_t_test_val)
        writer.writerow(["wilcox_test"] + all_statistics_wilcoxon_val)
        writer.writerow(["p_values for normality_cal"] + p_values_cal_val)
        writer.writerow(["p_values for normality_uncal"] + p_values_uncal_val)

    # Test data
    filename=f"aggregated_statistics_test_{dataset}_{model}.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["test"] + new_headers)
        writer.writerow(["paired_t_test"] + all_statistics_t_test_test)
        writer.writerow(["wilcox_test"] + all_statistics_wilcoxon_test)
        writer.writerow(["p_values for normality_cal"] + p_values_cal_test)
        writer.writerow(["p_values for normality_uncal"] + p_values_uncal_test)

def read_input(filename):
    return pd.read_excel(filename, sheet_name = None)

def read_brier_input_codeBERT(filename):
    brier_scores = [[], [], [], []]
    header = ["brier uncal", "brier_cal_inv_ps", "brier_cal_nps", "brier_cal_nts"]
    with open(filename, newline='') as f:
        index_brier_uncal = -1
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if index == 0:
                index_brier_uncal = row.index("brier_score_uncalibrated")
                continue
            brier_uncal = float(row[index_brier_uncal])
            brier_cal_inv_ps = float(row[index_brier_uncal+2])
            brier_cal_nps = float(row[index_brier_uncal+4])
            brier_cal_nts = float(row[index_brier_uncal+6])

            brier_scores[0].append(brier_uncal)
            brier_scores[1].append(brier_cal_inv_ps)
            brier_scores[2].append(brier_cal_nps)
            brier_scores[3].append(brier_cal_nts)
    brier_scores = np.asarray(brier_scores)
    brier_scores.astype('float64')
    return header, brier_scores


def read_input_codeBERT(filename):
    scores_cal = []
    scores_uncal = []
    header = []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if index == 0:
                header = row[1:]
                continue
            uncal_scores_row = row[1:13]
            uncal_scores_row = np.asarray(uncal_scores_row)
            uncal_scores_row = uncal_scores_row.astype('float64')
            scores_uncal.append(uncal_scores_row)

            cal_scores_row = row[13:]
            cal_scores_row = np.asarray(cal_scores_row)
            cal_scores_row = cal_scores_row.astype('float64')
            scores_cal.append(cal_scores_row)
    return header, np.asarray(scores_cal), np.asarray(scores_uncal)


def calculate_significance(uncalibrated_samples, calibrated_samples, all_statistics_t_test, all_statistics_wilcoxon, all_p_values):
    """
    Calculate significance for calibrated samples compared to uncalibrated sample either via t_test_paired or wilcoxon, depending on normality.

    Params:
        uncalibrated_samples (numpy.arr)
        calibrated_samples (numpy.arr)
        all_statistics_t_test (arr): array holding all repetitions, calculated values should be added here
        all_statistics_wilcoxon (arr): array holding all repetitions, calculated values should be added here
        all_p_values (arr): array holding tuples of p_values for normality
    """
    p_value_t_test = paired_t_test(uncalibrated_samples, calibrated_samples)
    p_value_wilcoxon = -1
    calibrated_samples_normal = is_normal_distributed(calibrated_samples, is_uncal=False)
    uncalibrated_samples_normal = is_normal_distributed(uncalibrated_samples, is_uncal=True)
    if not (calibrated_samples_normal[1] and uncalibrated_samples_normal[1]):
        print(f"[ERROR] not normally distributed")
        p_value_wilcoxon = wilcox_test(uncalibrated_samples, calibrated_samples)
        p_value_t_test = -1
    all_statistics_t_test.append(p_value_t_test)
    all_statistics_wilcoxon.append(p_value_wilcoxon)
    all_p_values.append((calibrated_samples_normal[0], uncalibrated_samples_normal[0]))


def run_calculations_LA_and_deepJIT(all_val_data, all_test_data, headers):
    """ Method used to orchestrate the calculations for all configurations for LApredict and DeepJIT data.
    """
    all_statistics_t_test_val = []
    all_statistics_t_test_test = []
    all_statistics_wilcoxon_val = []
    all_statistics_wilcoxon_test = []
    all_p_values_normality_val = []
    all_p_values_normality_test = []

    for index, header in enumerate(headers):
        uncal_samples_val = all_val_data[0][index]
        cal_platt_samples_val = all_val_data[1][index]
        cal_temp_samples_val = all_val_data[2][index]

        uncal_samples_test = all_test_data[0][index]
        cal_platt_samples_test = all_test_data[1][index]
        cal_temp_samples_test = all_test_data[2][index]

        print(f"[RUN Calc] header = {header} with index = {index} for VALIDATION FOLDS\nuncal = {uncal_samples_val}\ncal_platt = {cal_platt_samples_val}\ncal temp = {cal_temp_samples_val}\n\n FOR TEST: \nnucal = {uncal_samples_test}\ncal_platt = {cal_platt_samples_test}\ncal_temp = {cal_temp_samples_test}")

        print(f"[RUN CALC] uncal, cal_platt VALIDATION")
        calculate_significance(uncal_samples_val, cal_platt_samples_val, all_statistics_t_test_val, all_statistics_wilcoxon_val, all_p_values_normality_val)
        print(f"[RUN CALC] uncal, cal_temp VALIDATION")
        calculate_significance(uncal_samples_val, cal_temp_samples_val, all_statistics_t_test_val, all_statistics_wilcoxon_val, all_p_values_normality_val)

        print(f"[RUN CALC] uncal, cal_temp TEST")
        calculate_significance(uncal_samples_test, cal_platt_samples_test, all_statistics_t_test_test, all_statistics_wilcoxon_test, all_p_values_normality_test)
        print(f"[RUN CALC] uncal, cal_temp TEST")
        calculate_significance(uncal_samples_test, cal_temp_samples_test, all_statistics_t_test_test, all_statistics_wilcoxon_test, all_p_values_normality_test)

    return all_statistics_t_test_val, all_statistics_t_test_test, all_statistics_wilcoxon_val, all_statistics_wilcoxon_test, all_p_values_normality_val, all_p_values_normality_test

def aggregate_data_deepJIT(read_input, all_data, require_setup=False):
    if require_setup:
        # all_data = list(np.zeros((3,8,1)))
        all_data = []
        all_data.append([])
        all_data.append([])
        all_data.append([])
        for i in range(9):
            all_data[0].append([])
            all_data[1].append([])
            all_data[2].append([])

    headers = list(read_input['Platt'])[1:]
    print(f"[AGGREGATE DATA deepJIT] Headers = {headers} for all_data = {all_data}")
    for index, header in enumerate(headers):
        uncal_column = list(read_input[f'uncalib'][header])
        platt_column = list(read_input[f'Platt'][header])
        temp_column = list(read_input[f'Temp'][header])
        print(uncal_column, platt_column, temp_column)
        print(all_data[0][index], all_data[1][index], all_data[2][index])
        all_data[0][index] = all_data[0][index] + uncal_column
        all_data[1][index] += platt_column
        all_data[2][index] += temp_column
        print(f"[Aggregated Data deepJIT] Sanity Check header ({header}): \n uncal = {uncal_column}\n platt = {platt_column} \n temp = {temp_column}")
    return all_data

def aggregate_data_LApredict(read_input):
    all_validation_data = []
    all_test_data = []
    headers = list(read_input['Sheet1'])[1:9]
    headers.append(list(read_input['Sheet1'])[10])
    print(f"Headers = {headers}")
    for header in headers:
        validation_data = []
        test_data = []
        # number of sheets for LApredict
        for i in range(1,21):
            print(f"i={i}")
            column_val = list(read_input[f'Sheet{i}'][header][0:10])
            column_test = read_input[f'Sheet{i}'][header][10]
            print(f"col_val = {column_val} and test = {column_test}, {read_input[f'Sheet{i}'][header][10]}")
            validation_data+=column_val
            test_data.append(column_test)
        all_validation_data.append(validation_data)
        all_test_data.append(test_data)
        print(f"[Aggregated Data] Sanity Check:\n Validation Data for header {header} is:\n{validation_data}\nTest Data for header {header} is:\n{test_data}")
    return all_validation_data, all_test_data


def statistic(x, axis):
    """
    Taken from https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_normaltest.html
    """
    # Get only the `normaltest` statistic; ignore approximate p-value
    # still requires n >= 20
    return stats.normaltest(x, axis=axis).statistic


def is_normal_distributed_mc(samples, is_uncal, required_p_value=0.05):
    """
    Caclualte normality based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.monte_carlo_test.html#scipy.stats.monte_carlo_test, example from https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_normaltest.html

    Functioning is similar to other func: H_0 = is normally distributed, p_value <= 0.05 significant rejection of H_0
    """
    calibration = "uncalibrated" if is_uncal else "calibrated"
    res = stats.monte_carlo_test(samples, stats.norm.rvs, statistic)
    print(f"REsult = {res}")
    if res.pvalue < required_p_value:
        print(f"[NORMAL DISTRIBUTION MC] Likely that {calibration} distribution is not normal with p_value of {res.pvalue}")
        return (res.pvalue, False)
    print(f"[NORMAL DISTRIBUTION MC] for {calibration} p-value = {res.pvalue}")
    return (res.pvalue, True)


def is_normal_distributed_dagostino(samples, is_uncal, required_p_value=0.05):
    """
    Method used to determine if samples are drawn from a normal distribution.
    H_0_ = it is normally distributed
    Hence, a low p_value can be seen as evidence, that this sample distribution is not drawn from a normal distribution.
    Sufficient "high" p_value currently (randomly) set as 0.2.
    Noteworthy: only works properly for at least 50 samples.

    More infos https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_normaltest.html#hypothesis-normaltest
    """
    res = stats.normaltest(samples)
    calibration = "uncalibrated" if is_uncal else "calibrated"
    if res.pvalue < required_p_value:
        print(f"[NORMAL DISTRIBUTION] Likely that {calibration} distribution is not normal with p_value of {res.pvalue}")
        return (res.pvalue, False)
    print(f"[NORMAL DISTRIBUTION] statistic for {calibration} distribution: {res.statistic}\np-value = {res.pvalue}")
    return (res.pvalue, True)

def is_normal_distributed(samples, is_uncal, required_p_value=0.05):
    if len(samples) >= 50:
        print(f"[NORMAL DISTRIBUTION] Sample size >= 50 using default normality for is_uncal = {is_uncal}")
        return is_normal_distributed_dagostino(samples, is_uncal, required_p_value)
    print(f"[NORMAL DISTRIBUTION] Sample size < 50 using monte carlo normality for is uncal = {is_uncal}")
    return is_normal_distributed_mc(samples, is_uncal=False, required_p_value=0.05)

def paired_t_test(samples_uncal, samples_cal):
    """
    Compute the paired t test for the uncalibrated and calibrated ECE scores.
    This assumes H_0_ = identical average expected values for both samples

    More infos: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    """
    result = stats.ttest_rel(samples_uncal, samples_cal)
    print(f"[PAIRED T_TEST]] results = {result}\n with conf interval = {result.confidence_interval(0.95)}")
    return result.pvalue

def wilcox_test(uncalibrated_samples, calibrated_samples):
    """ TODO: Question of rounding for proper analysis -> else simply uncal - cal will be used. This might lead to some weird behaviour, as due to floating inacurracy the some numbers are treated (un)equally, that should not.
    H_0: samples are from same distribution 

    More infos: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    """
    res = stats.wilcoxon(uncalibrated_samples, calibrated_samples)
    print(f"[WILCOXON]] results = {res}\n")
    return res.pvalue

def process_brier_inputs_codeBERT(headers, brier_scores):
    all_statistics_t_test = []
    all_statistics_wilcoxon = []
    all_p_values = []
    for index, header in enumerate(headers):
        # skip uncalibrated
        if index == 0: continue
        calibrated_samples = np.asarray( brier_scores[index])
        uncalibrated_samples = np.asarray(brier_scores[0])
        assert len(calibrated_samples) == len(uncalibrated_samples) == 100
        print(f"[Process Brier Inputs CodeBERT] Sanity Check for header {header}: calibrated_samples = {calibrated_samples[:3]}, uncalibrated_samples = {uncalibrated_samples[:3]}")
        print(f"uncal = {uncalibrated_samples}")
        print(f"cal = {calibrated_samples}")
        calculate_significance(uncalibrated_samples, calibrated_samples, all_statistics_t_test, all_statistics_wilcoxon, all_p_values)
    return all_statistics_t_test, all_statistics_wilcoxon, all_p_values, headers
        

def process_inputs_codeBERT(read_input, configs_per_calibration=12):
    header, scores_cal, scores_uncal = read_input
    number_of_calibrations = len(scores_cal[0])//configs_per_calibration
    # print(f"Number of calibrations = {number_of_calibrations}")
    all_statistics_t_test = []
    all_statistics_wilcoxon = []
    all_p_values = []
    new_header = []
    for i in range(number_of_calibrations):
        for j in range(configs_per_calibration):
            # Gathering comparable samples
            offset = i*configs_per_calibration+j
            calibrated_samples = scores_cal[:,offset]
            uncalibrated_samples = scores_uncal[:,j]
            print(f"[COMPARING] column = {header[offset+configs_per_calibration]} to column = {header[j]} \nwith indices i= {i} and j={j} offset = {offset}")
            print(f"calibrated_samples = {calibrated_samples[0:3]} with len={len(calibrated_samples)}")
            print(f"uncalibrated samples = {uncalibrated_samples[0:3]} with len = {len(uncalibrated_samples)}")

            # Calculate significance based on normality
            calculate_significance(uncalibrated_samples, calibrated_samples, all_statistics_t_test, all_statistics_wilcoxon, all_p_values)
            new_header.append(f"{header[j]} vs {header[offset+configs_per_calibration]}")
            print("\n")

    return all_statistics_t_test, all_statistics_wilcoxon, new_header, all_p_values




def driver_deepJIT(datasets):
    calibrations = ['uncalib', 'Platt', 'Temp']
    reps = 5
    for dataset in datasets:
        all_data_test = []
        all_data_val = []
        is_first_run = True
        if dataset == 'qt':
            reps = 3
            all_data_test = []
            all_data_val = []
        for index in range(reps):
            test_filename = f"./deepjit/{dataset}/test_data/AGG_calibration_scores_on_uncalib_PlattS_TempS_test_data_run{index}.xlsx"
            validation_filename = f"./deepjit/{dataset}/validation_data/AGG_calibration_scores_on_uncalib_PlattS_TempS_valid_data_run{index}.xlsx"
            print(f"[DRIVER deepJIT] read file validation = {validation_filename}\n read file test = {test_filename}")
            read_input_test = read_input(test_filename)
            read_input_val = read_input(validation_filename)
            if index > 0: is_first_run = False
            print(f"[DRIVER DEEPJIT] aggregate value")
            all_data_val = aggregate_data_deepJIT(read_input_val, all_data_val, is_first_run)
            print(f"[DRIVER DEEPJIT] aggregate test")
            all_data_test = aggregate_data_deepJIT(read_input_test, all_data_test, is_first_run)

        print(f"[DRIVER deepJIT] run_calc for {dataset} for validation folds uncalibrated = {all_data_val[0]}\n Platt = {all_data_val[1]}\n Temp = {all_data_val[2]}")
        print(f"[DRIVER deepJIT] run_calc for {dataset} for test values uncalibrated = {all_data_test[0]}\n Platt = {all_data_test[1]}\n Temp = {all_data_test[2]}")
        headers = list(read_input_val['Platt'])[1:]
        assert len(all_data_val[0]) == 9
        assert len(all_data_test[0]) == 9
        assert len(all_data_val[1]) == 9
        assert len(all_data_test[1]) == 9
        assert len(all_data_val[2]) == 9
        assert len(all_data_test[2]) == 9

        assert len(all_data_val[0][0]) == 10*reps
        assert len(all_data_val[1][0]) == 10*reps
        assert len(all_data_val[2][2]) == 10*reps

        assert len(all_data_test[0][0]) == 10*reps
        assert len(all_data_test[1][1]) == 10*reps
        assert len(all_data_test[2][7]) == 10*reps
        all_statistics_t_test_val, all_statistics_t_test_test, all_statistics_wilcoxon_val, all_statistics_wilcoxon_test, all_p_values_normality_val, all_p_values_normality_test = run_calculations_LA_and_deepJIT(all_data_val, all_data_test, headers)
        write_LApredict_and_deepJIT(all_statistics_t_test_val, all_statistics_t_test_test, all_statistics_wilcoxon_val, all_statistics_wilcoxon_test, all_p_values_normality_val, all_p_values_normality_test, headers, dataset, model="deepJIT")


def driver_LA_predict(datasets):
    calibrations = ['uncalibrated', 'PlattScaling', 'TempScale']
    all_val_data = []
    all_test_data = []
    for dataset in datasets:
        for index, calibration in enumerate(calibrations):
            filename = f"./lapredict/LA_predict_New_Runs/valid_and_testSet_{calibration}_{dataset}.xlsx"
            print(f"[READ Input] file = {filename}")
            input_data_LApredict = read_input(filename)
            val_data, test_data = aggregate_data_LApredict(input_data_LApredict)
            all_val_data.append(val_data)
            all_test_data.append(test_data)
        print(f"[DRIVER LApredict] run_calc for {dataset} for validation folds uncalibrated = {all_val_data[0]}\n Platt = {all_val_data[1]}\n Temp = {all_val_data[2]}")
        print(f"[DRIVER LApredict] run_calc for {dataset} for test values uncalibrated = {all_test_data[0]}\n Platt = {all_test_data[1]}\n Temp = {all_test_data[2]}")
        headers = list(input_data_LApredict['Sheet1'])[1:9]
        headers.append(list(input_data_LApredict['Sheet1'])[10])
        assert len(all_val_data[0]) == 9
        assert len(all_test_data[0]) == 9

        assert len(all_val_data[0][0]) == 200
        assert len(all_val_data[1][0]) == 200
        assert len(all_val_data[2][2]) == 200

        assert len(all_test_data[0][0]) == 20
        assert len(all_test_data[1][1]) == 20
        assert len(all_test_data[2][7]) == 20
        all_statistics_t_test_val, all_statistics_t_test_test, all_statistics_wilcoxon_val, all_statistics_wilcoxon_test, all_p_values_normality_val, all_p_values_normality_test = run_calculations_LA_and_deepJIT(all_val_data, all_test_data, headers)
        write_LApredict_and_deepJIT(all_statistics_t_test_val, all_statistics_t_test_test, all_statistics_wilcoxon_val, all_statistics_wilcoxon_test, all_p_values_normality_val, all_p_values_normality_test, headers, dataset)


def drive_appending_brier_codeBERT(datasets):
    for dataset in datasets:
        exp_results = "experiment_results" if dataset == 'op' else "experiment_results_qt"
        filename = f"../{exp_results}/Calibrated_Predictions_Metrics_evaluation_{dataset}.csv"
        header, brier_input = read_brier_input_codeBERT(filename)
        statistics_t_test, statistics_wilcox, all_p_values_normal, header = process_brier_inputs_codeBERT(header, brier_input)
        write_statistics(statistics_t_test, statistics_wilcox, header[1:], all_p_values_normal, dataset=dataset, model="CodeBERT", is_brier_only=True)


def main(should_run_codeBERT, should_run_LApredict, should_run_deepJIT, should_append_to_codeBERT):
    """ Script used to calculate the significance of the MSR 2025 Paper "On the calibration of Just-in-time Defect Prediction" results. 
    
    """
    datasets = ['op', 'qt']
    if should_run_deepJIT:
        driver_deepJIT(datasets)

    if should_run_LApredict:
        driver_LA_predict(datasets)

    if should_run_codeBERT:
        for dataset in datasets:
            filename = f"./aggregated_ece_calibration_False_{dataset}.csv"
            input_data_codebert = read_input_codeBERT(filename)
            statistics_t_test, statistics_wilcox, header, all_p_values_normal = process_inputs_codeBERT(input_data_codebert)
            write_statistics(statistics_t_test, statistics_wilcox, header, all_p_values_normal, dataset=dataset, model="CodeBERT")

    if should_append_to_codeBERT:
        drive_appending_brier_codeBERT(datasets)


if __name__ == '__main__':
    main(should_run_codeBERT=True, should_run_LApredict=True, should_run_deepJIT=True, should_append_to_codeBERT=True)
