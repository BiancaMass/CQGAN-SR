
def destination_qubit_index_calculator(original_rows_num, original_cols_num):
    destination_cols = original_cols_num * 2 + 1
    destination_qubit_indexes = []
    for r in range(1, original_rows_num + 1):
        for c in range(1, original_cols_num + 1):
            original_i, original_j = r, c
            destination_i, destination_j = original_i*2, original_j*2
            destination_index = (destination_i - 1) * destination_cols + destination_j
            destination_qubit_indexes.append(destination_index)

    return destination_qubit_indexes
