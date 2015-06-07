from hyperopt import hp


def get_search_space():

    max_kernel_norm_mu = 1.9365
    max_kernel_norm_sigma = 1
    kernel_shape_width = 0.1
    kernel_shape_height = 0.5
    kernel_stride_width = 1
    kernel_stride_height = 1
    pool_shape_width = .1
    pool_shape_height = .1
    pool_stride_width = 1
    pool_stride_height = 1
    irange_mean = 0.05
    irange_dev = 0.03
    relu_left_slope_mean = 0    # to juz na pewno jest zle...
    relu_left_slope_sigma = 0.1

    space = [
        {
            'h0': hp.choice('first layer', [
                {
                    'layer type': 'ConvRectifiedLinear',
                    'output channels': hp.randint('output channels', 40), # zmienic na cos madrzejszego
                    'max kernel norm': hp.normal('max kernel norm', max_kernel_norm_mu, max_kernel_norm_sigma),
                    'pool shape width': hp.uniform('pool shape width', 0, pool_shape_width),
                    'pool shape height': hp.uniform('pool shape height', 0, pool_shape_height),
                    'pool stride width': hp.uniform('pool stride width', 0, pool_stride_width),
                    'pool stride height': hp.uniform('pool stride height', 0, pool_stride_height),
                    'kernel shape width': hp.uniform('kernel shape width', 0, kernel_shape_width),
                    'kernel shape height': hp.uniform('kernel shape height', 0, kernel_shape_height),
                    'kernel stride width': hp.uniform('kernel stride width', 0, kernel_stride_width),
                    'kernel stride height': hp.uniform('kernel stride height', 0, kernel_stride_height),
                    'irange': hp.normal('irange', irange_mean, irange_dev),
                    'border mode': hp.choice('border mode', ['full', 'valid'])
                },
                {
                    'layer type': 'ConvElementWise',
                    'output channels': hp.randint('output channels', 40), # zmienic na cos madrzejszego
                    'max kernel norm': hp.normal('max kernel norm', max_kernel_norm_mu, max_kernel_norm_sigma),
                    'nonlinearity': hp.choice('nonlinearity', [
                        {
                            'nonlinearity type': 'RectifierConvNonlinearity',
                            'left slope': hp.normal('left slope of relu', relu_left_slope_mean, relu_left_slope_sigma)
                        },
                        {
                            'nonlinearity type': 'TanhConvNonlinearity'
                        },
                        {
                            'nonlinearity type': 'SigmoidConvNonlinearity'
                        }
                    ]),
                    'pool shape width': hp.uniform('pool shape width', 0, pool_shape_width),
                    'pool shape height': hp.uniform('pool shape height', 0, pool_shape_height),
                    'pool stride width': hp.uniform('pool stride width', 0, pool_stride_width),
                    'pool stride height': hp.uniform('pool stride height', 0, pool_stride_height),
                    'kernel shape width': hp.uniform('kernel shape width', 0, kernel_shape_width),
                    'kernel shape height': hp.uniform('kernel shape height', 0, kernel_shape_height),
                    'kernel stride width': hp.uniform('kernel stride width', 0, kernel_stride_width),
                    'kernel stride height': hp.uniform('kernel stride height', 0, kernel_stride_height),
                    'irange': hp.normal('irange', irange_mean, irange_dev),
                    'border mode': hp.choice('border mode', ['full', 'valid'])
                },
            ]),
        },
        {
            'h1': hp.choice('second layer', [
                {
                    'layer type': 'ConvRectifiedLinear',
                    'output channels': hp.randint('output channels', 20), # zmienic na cos madrzejszego
                    'max kernel norm': hp.normal('max kernel norm', max_kernel_norm_mu, max_kernel_norm_sigma),
                    'pool shape width': hp.uniform('pool shape width', 0, pool_shape_width),
                    'pool shape height': hp.uniform('pool shape height', 0, pool_shape_height),
                    'pool stride width': hp.uniform('pool stride width', 0, pool_stride_width),
                    'pool stride height': hp.uniform('pool stride height', 0, pool_stride_height),
                    'kernel shape width': hp.uniform('kernel shape width', 0, kernel_shape_width),
                    'kernel shape height': hp.uniform('kernel shape height', 0, kernel_shape_height),
                    'kernel stride width': hp.uniform('kernel stride width', 0, kernel_stride_width),
                    'kernel stride height': hp.uniform('kernel stride height', 0, kernel_stride_height),
                    'irange': hp.normal('irange', irange_mean, irange_dev),
                },
                {
                    'layer type': 'ConvElementWise',
                    'output channels': hp.randint('output channels', 20), # zmienic na cos madrzejszego
                    'max kernel norm': hp.normal('max kernel norm', max_kernel_norm_mu, max_kernel_norm_sigma),
                    'nonlinearity': hp.choice('nonlinearity', [
                        {
                            'nonlinearity type': 'RectifierConvNonlinearity',
                            'left slope': hp.normal('left slope of relu', relu_left_slope_mean, relu_left_slope_sigma)
                        },
                        {
                            'nonlinearity type': 'TanhConvNonlinearity'
                        },
                        {
                            'nonlinearity type': 'SigmoidConvNonlinearity'
                        },
                    ]),
                    'pool shape width': hp.uniform('pool shape width', 0, pool_shape_width),
                    'pool shape height': hp.uniform('pool shape height', 0, pool_shape_height),
                    'pool stride width': hp.uniform('pool stride width', 0, pool_stride_width),
                    'pool stride height': hp.uniform('pool stride height', 0, pool_stride_height),
                    'kernel shape width': hp.uniform('kernel shape width', 0, kernel_shape_width),
                    'kernel shape height': hp.uniform('kernel shape height', 0, kernel_shape_height),
                    'kernel stride width': hp.uniform('kernel stride width', 0, kernel_stride_width),
                    'kernel stride height': hp.uniform('kernel stride height', 0, kernel_stride_height),
                    'irange': hp.normal('irange', irange_mean, irange_dev),
                },
                None
            ]),
        },
        {
            'layer type': 'softmax',
            'irange': hp.normal('irange', irange_mean, irange_dev),

        },
        {'fajanse:': 'momentum adjustor and so on ...'}     # TODO: implement
        ]

    return space





