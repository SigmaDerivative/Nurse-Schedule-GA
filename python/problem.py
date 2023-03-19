from problem_setup import Problem

# PROBLEM SETUP
problem = Problem("data/train_0.json")
nbr_nurses = problem.nbr_nurses
nbr_patients = problem.nbr_patients
travel_times = problem.travel_times
capacity_nurse = problem.capacity_nurse
patients = problem.numpy_patients
list_patients = problem.list_patients
coordinates = patients[:, :2]

start_after_end_penalty = 3
end_after_end_penalty = 3
capacity_penalty = 3
