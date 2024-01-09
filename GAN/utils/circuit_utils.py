def destination_qubit_index_calculator(original_rows_num, original_cols_num, scaling_factor):
    """
    Computes destination qubit indexes for scaled image dimensions. Each pixel becomes a s*s
    square where s is the scaling factor. The initial pixel is located in the top left corner of
    the square. Future implementations might consider moving the pixel in the middle, which works
    especially well for odd numbers of s.

    Args:
        original_rows_num (int): Number of rows in the original image.
        original_cols_num (int): Number of columns in the original image.
        scaling_factor (int): Factor by which the image is scaled (same for both dims).

    Returns:
        List[int]: List of destination qubit indexes after scaling.

        """
    destination_cols = original_cols_num * scaling_factor
    destination_qubit_indexes = []
    for r in range(original_rows_num):
        for c in range(original_cols_num):
            destination_i, destination_j = r * scaling_factor, c * scaling_factor
            destination_index = destination_i * destination_cols + destination_j
            destination_qubit_indexes.append(destination_index)

    return destination_qubit_indexes
