# alpha_spectrum = epochs.compute_psd(method='multitaper', fmin=freq_bands['alpha'][0], fmax=freq_bands['alpha'][1], tmin=0, tmax=None)
# beta_spectrum = epochs.compute_psd(method='multitaper', fmin=freq_bands['beta'][0], fmax=freq_bands['beta'][1], tmin=0, tmax=None)
# num_of_epochs = config.epoch_information['duration']

# df_alpha = alpha_spectrum.get_data().reshape(int(config.epoch_information['duration']/bandpower_epoch_duration),
#                                              len(channel_names),
#                                              -1)
# df_alpha.columns = alpha_spectrum.freqs
#
# df_beta = beta_spectrum.get_data().reshape(int(config.epoch_information['duration']/bandpower_epoch_duration),
#                                            len(channel_names),
#                                            -1)
# df_beta.columns = beta_spectrum.freqs


# print(psd)