import pandas as pd
import numpy as np
import scipy
from brainflow.data_filter import DataFilter

def identify_artifacts_by_kurtosis(sources, threshold=5):
    kurtosis_values = np.apply_along_axis(scipy.stats.kurtosis, axis=1, arr=sources)
    artifact_indices = np.where(kurtosis_values > threshold)[0]
    return artifact_indices

def algo_1(df, start_index, end_index):
    # print(df.iloc[start_index:end_index])
    # print(len(df.iloc[start_index:end_index]))

    data = np.ascontiguousarray(df)

    unmixing_matrix, sources, mixing_matrix, _ = DataFilter.perform_ica(data, 8)

    artifacts_indices = identify_artifacts_by_kurtosis(sources)

    cleaned_sources = np.copy(sources)
    cleaned_sources[artifacts_indices, :] = 0

    cleaned_data = unmixing_matrix @ cleaned_sources
    cleaned_df = pd.DataFrame(cleaned_data.T, columns=df.columns)

    return cleaned_df
