def transfer(x_data_un, x_data_su, y_data, pad_size, handle,
             num_epochs=200,
             batch_size=1,
             filters=100,
             filter_size=10,
             validation=0.2,
             holdout=.1,
             rate=.01,
             model=None):
    net_arch = nets.CoDER(pad_size=pad_size, filters=filters, filter_size=filter_size)
    # net_arch = nets.CoDER_test(pad_size, filters, filter_size)
    handle.model = net_arch.handle

    conv_unsuper = DeepTrainer(net_arch,
                               batch_size=batch_size,
                               learning_rate=rate)

    conv_unsuper.display_network_info()

    conv_unsuper.fit(x_data_un, x_data_un, num_epochs, validation_split=validation, holdout=holdout)
    conv_unsuper.save_train_history(handle)
    conv_unsuper.save_model_to_file(handle)

    # Extract the motifs from the convolutional layers
    motif_extraction(conv_unsuper.custom_fun(), x_data_un, handle, filters, filter_size)

    print('Transfering parameters to supervised learning network')
    net_arch = nets.CoHST(pad_size=pad_size, filters=filters, filter_size=filter_size)

    handle.model = handle.model + net_arch.handle

    conv_super = DeepTrainer(net_arch,
                             batch_size=batch_size,
                             classification=True,
                             learning_rate=.01)

    conv_super.display_network_info()
    conv_super.set_conv_param_values(conv_unsuper.get_conv_param_values())

    conv_super.fit(x_data_su, y_data, num_epochs, validation_split=validation, holdout=holdout)
    conv_super.save_train_history(handle)
    conv_super.save_model_to_file(handle)

    # motif_extraction(conv_super.custom_fun(), x_data_un, handle, filters, filter_size)


def supervised(x_data, y_data, pad_size, handle,
               epochs=200,
               batch_size=1,
               filters=100,
               filter_size=10,
               validation=0.2,
               holdout=.1,
               rate=.01,
               model=None):
    net_arch = nets.CoHST(pad_size=pad_size, filters=filters, filter_size=filter_size)

    handle.model = net_arch.handle

    conv_net = DeepTrainer(net_arch,
                           batch_size=batch_size,
                           classification=True,
                           learning_rate=rate)
    conv_net.display_network_info()
    if model:
        conv_net.set_all_param_values(model)
        conv_net.load_train_history(model.split('.')[0])
    conv_net.fit(x_data, y_data, epochs, validation_split=validation, holdout=holdout)
    conv_net.save_train_history(handle)
    conv_net.save_model_to_file(handle)

    # motif_extraction(conv_net.custom_fun(), x_data, handle)
