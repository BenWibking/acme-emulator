from pipeline_defs import *
from pathlib import Path

param_files = param_files_in_dir(param_dir)

def task_emulate_wp_test():
    """emulate wp for all param_files"""

    for input_param_file in param_files:
        script = "./Analysis/emulate_wp.py"
        fiducial_param_file = "Params/LOWZ/NHOD_fiducial.template_param"
        fiducial_bg_file = "Emulator/fiducial_ln_bg.txt"
        fiducial_bnl_file = "Emulator/fiducial_ln_bnl.txt"
        deriv_files_string = "--deriv_bg_file Emulator/ln_bg_alpha.txt alpha --deriv_bg_file Emulator/ln_bg_del_gamma.txt del_gamma --deriv_bg_file Emulator/ln_bg_M1_over_Mmin.txt M1_over_Mmin --deriv_bg_file Emulator/ln_bg_ngal.txt ngal --deriv_bg_file Emulator/ln_bg_Omega_M.txt Omega_M --deriv_bg_file Emulator/ln_bg_sigma_8.txt sigma_8 --deriv_bg_file Emulator/ln_bg_q_env.txt q_env --deriv_bg_file Emulator/ln_bg_siglogM.txt siglogM --deriv_bnl_file Emulator/ln_bnl_Omega_M.txt Omega_M --deriv_bnl_file Emulator/ln_bnl_sigma_8.txt sigma_8 --xi_mm_linear Emulator/linear_theory_matter.autocorr"
        deps = [script, fiducial_param_file, fiducial_bg_file, fiducial_bnl_file]
        targets = [txt_emulated_wp_this_param(input_param_file)] # Params/LOWZ_HOD_tests/NHOD_test.1.template_param.emulated_wp

        yield {
            'actions': ["python %(script)s %(fiducial_param)s %(fiducial_bg)s %(fiducial_bnl)s %(input_param)s %(target)s %(deriv_files)s"
                        % {"script": script, "target": targets[0], "fiducial_param": fiducial_param_file,
                           "fiducial_bg": fiducial_bg_file, "fiducial_bnl": fiducial_bnl_file,
                           "input_param": input_param_file, "deriv_files": deriv_files_string}],
            'file_dep': deps,
            'task_dep': [],
            'targets': targets,
            'name': str(input_param_file),
        }

def task_fractional_accuracy_emulated_wp():
    """compute (emulated wp - true wp)/(true wp) for all param_files"""

    for input_param_file in param_files:
        script = "./Analysis/compute_accuracy.py"
        true_wp_file = txt_reconstructed_wp_this_param(input_param_file)
        emulated_wp_file = txt_emulated_wp_this_param(input_param_file)
        deps = [script, true_wp_file, emulated_wp_file]
        targets = [txt_fractional_accuracy_emulated_wp_this_param(input_param_file)]

        yield {
            'actions': ["python %(script)s %(true_wp)s %(emulated_wp)s %(target)s"
                        % {"script": script, "target": targets[0],
                           "true_wp": true_wp_file,
                           "emulated_wp": emulated_wp_file, }],
            'file_dep': deps,
            'task_dep': [],
            'targets': targets,
            'name': str(input_param_file),
        }

def task_emulate_DS_test():
    """emulate DS for all param_files"""

    for input_param_file in param_files:
        script = "./Analysis/emulate_DeltaSigma.py"
        fiducial_param_file = "Params/LOWZ/NHOD_fiducial.template_param"
        fiducial_bg_file = "Emulator/fiducial_ln_bg.txt"
        fiducial_bnl_file = "Emulator/fiducial_ln_bnl.txt"
        fiducial_rgm_file = "Emulator/fiducial_ln_rgm.txt"
        deriv_files_string = "--deriv_bg_file Emulator/ln_bg_alpha.txt alpha --deriv_bg_file Emulator/ln_bg_del_gamma.txt del_gamma --deriv_bg_file Emulator/ln_bg_M1_over_Mmin.txt M1_over_Mmin --deriv_bg_file Emulator/ln_bg_ngal.txt ngal --deriv_bg_file Emulator/ln_bg_Omega_M.txt Omega_M --deriv_bg_file Emulator/ln_bg_sigma_8.txt sigma_8 --deriv_bg_file Emulator/ln_bg_q_env.txt q_env --deriv_bg_file Emulator/ln_bg_siglogM.txt siglogM --deriv_bnl_file Emulator/ln_bnl_Omega_M.txt Omega_M --deriv_bnl_file Emulator/ln_bnl_sigma_8.txt sigma_8 --xi_mm_linear Emulator/linear_theory_matter.autocorr --deriv_rgm_file Emulator/ln_rgm_alpha.txt alpha --deriv_rgm_file Emulator/ln_rgm_del_gamma.txt del_gamma --deriv_rgm_file Emulator/ln_rgm_M1_over_Mmin.txt M1_over_Mmin --deriv_rgm_file Emulator/ln_rgm_ngal.txt ngal --deriv_rgm_file Emulator/ln_rgm_Omega_M.txt Omega_M --deriv_rgm_file Emulator/ln_rgm_sigma_8.txt sigma_8 --deriv_rgm_file Emulator/ln_rgm_q_env.txt q_env --deriv_rgm_file Emulator/ln_rgm_siglogM.txt siglogM"
        deps = [script, fiducial_param_file, fiducial_bg_file, fiducial_bnl_file, fiducial_rgm_file]
        targets = [txt_emulated_DS_this_param(input_param_file)]

        yield {
            'actions': ["python %(script)s %(fiducial_param)s %(fiducial_bg)s %(fiducial_bnl)s %(fiducial_rgm)s %(input_param)s %(target)s %(deriv_files)s"
                        % {"script": script, "target": targets[0], "fiducial_param": fiducial_param_file,
                           "fiducial_bg": fiducial_bg_file, "fiducial_bnl": fiducial_bnl_file,
                           "fiducial_rgm": fiducial_rgm_file,
                           "input_param": input_param_file, "deriv_files": deriv_files_string}],
            'file_dep': deps,
            'task_dep': [],
            'targets': targets,
            'name': str(input_param_file),
        }

def task_fractional_accuracy_emulated_DS():
    """compute (emulated DS - true DS)/(true DS) for all param_files"""

    for input_param_file in param_files:
        script = "./Analysis/compute_accuracy.py"
        true_DS_file = txt_reconstructed_DeltaSigma_this_param(input_param_file)
        emulated_DS_file = txt_emulated_DS_this_param(input_param_file)
        deps = [script, true_DS_file, emulated_DS_file]
        targets = [txt_fractional_accuracy_emulated_DS_this_param(input_param_file)]

        yield {
            'actions': ["python %(script)s %(true_DS)s %(emulated_DS)s %(target)s"
                        % {"script": script, "target": targets[0],
                           "true_DS": true_DS_file,
                           "emulated_DS": emulated_DS_file, }],
            'file_dep': deps,
            'task_dep': [],
            'targets': targets,
            'name': str(input_param_file),
        }
