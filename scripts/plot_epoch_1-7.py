import numpy as np
from matplotlib import pyplot as plt


"""
These numbers were (painfully) extracted and formatted from the keras logs
under logs/. 

Epochs 1-7 are in one file, after which we added l2 reg, and then
did epochs 8-20.
"""

epoch1 = {
    'x' : np.array([1., 100., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [378364.0312, 119044.3281, 80540.8672, 72485.3631, 68270.9631, 65601.2418, 63880.2223, 62391.8667, 61146.4818, 60260.3811],
    'iou' : [0.1665, 0.4204, 0.5703, 0.6061, 0.6268, 0.6396, 0.6475, 0.6538, 0.6596, 0.6636],
    'vloss' : 61352.2011,
    'viou' : 0.6909
}

epoch2 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [61078.7598, 52234.4410, 51989.6470, 51951.3594, 51668.4240, 51090.3679, 50954.0791, 50795.9881, 50681.1486],
    'iou' : [0.6429, 0.7076, 0.7045, 0.7046, 0.7053, 0.7071, 0.7071, 0.7074, 0.7087],
    'vloss' : 52981.4480,
    'viou' : 0.7031
}

epoch3 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [28614.7698, 48403.9542, 48540.6464, 48439.8972, 48378.9897, 48316.0248, 48048.8650, 47823.2336, 47693.9662],
    'iou' : [0.6233, 0.7195, 0.7211, 0.7206, 0.7203, 0.7209, 0.7220, 0.7224, 0.7230],
    'vloss' : 52984.8885,
    'viou' : 0.7056
}

epoch4 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [55985.7891, 46152.6722, 45841.4347, 46056.1062, 45911.6120, 45874.9179, 45910.3057, 45929.5863, 45995.2859],
    'iou' : [0.7498, 0.7274, 0.7283, 0.7287, 0.7304, 0.7305, 0.7319, 0.7318, 0.7318],
    'vloss' : 50369.4005,
    'viou' : 0.7146
}

epoch5 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [39211.5176, 44956.1495, 44670.8546, 44710.1773, 44631.8413, 44405.8628, 44531.0005, 44549.8002, 44531.7530],
    'iou' : [0.6444, 0.7401, 0.7387, 0.7388, 0.7396, 0.7401, 0.7392, 0.7393, 0.7399],
    'vloss' : 50258.8032,
    'viou' : 0.7263
}

epoch6 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [43040.3330, 43801.6961, 43831.4814, 43722.3142, 43756.1244, 43829.7247, 43847.7397, 43779.8761, 43588.0624],
    'iou' : [0.7293, 0.7425, 0.7449, 0.7447, 0.7437, 0.7437, 0.7436, 0.7436, 0.7443],
    'vloss' : 47892.2107,
    'viou' : 0.7377
}

epoch7 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [38539.5698, 42349.6251, 42251.4041, 42428.7473, 42233.3160, 42358.8873, 42499.2086, 42466.9905, 42375.9954],
    'iou' : [0.7671, 0.7528, 0.7494, 0.7487, 0.7506, 0.7500, 0.7497, 0.7499, 0.7505],
    'vloss' : 47624.5262,
    'viou' : 0.7310
}

epoch8 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [],
    'iou' : [],
    'vloss' : 1,
    'viou' : 1
}

epoch9 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [],
    'iou' : [],
    'vloss' : 1,
    'viou' : 1
}

epoch10 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [],
    'iou' : [],
    'vloss' : 1,
    'viou' : 1
}

epoch11 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [],
    'iou' : [],
    'vloss' : 1,
    'viou' : 1
}

epoch12 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [],
    'iou' : [],
    'vloss' : 1,
    'viou' : 1
}

epoch13 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [],
    'iou' : [],
    'vloss' : 1,
    'viou' : 1
}

epoch14 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [],
    'iou' : [],
    'vloss' : 1,
    'viou' : 1
}

epoch15 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [],
    'iou' : [],
    'vloss' : 1,
    'viou' : 1
}

epoch16 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [],
    'iou' : [],
    'vloss' : 1,
    'viou' : 1
}

epoch17 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [],
    'iou' : [],
    'vloss' : 1,
    'viou' : 1
}

epoch18 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [],
    'iou' : [],
    'vloss' : 1,
    'viou' : 1
}

epoch19 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [],
    'iou' : [],
    'vloss' : 1,
    'viou' : 1
}

epoch20 = {
    'x' : np.array([1., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.]) / 8014.,
    'loss' : [],
    'iou' : [],
    'vloss' : 1,
    'viou' : 1
}

def printl(obj, name):
    print("%s: %s" % (name, str(obj)))


def plot_data(x, loss, iou, savepath=None):
    """ If no savepath, save in curr dir.
    """

    # Training
    fig, ax1 = plt.subplots()
    
    ax1.plot(x, loss, 'b-')
    ax1.set_xlabel('Number of completed epochs')
    ax1.set_ylabel('Total cross entropy loss', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(x, iou, 'g-')
    ax2.set_ylabel('IOU', color='g')
    ax2.tick_params('y', colors='g')

    fig.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        from datetime import datetime
        plt.savefig("figure_%s.png" % str(datetime.now()))



if __name__ == "__main__":
    local_vars = dict(locals()) # shallow copy to avoid concurrency problems
    epochs = [(int(v[5:]), local_vars[v]) for v in local_vars if 'epoch' in v]
    epochs = sorted(epochs, key=lambda tup: tup[0])

    x = []
    loss = []
    iou = []

    x_eps = list(range(len(epochs)))
    vloss_eps = []
    viou_eps = []
    
    for i, ep in epochs:
        x.extend(list(ep['x'] + (i-1) )) # So epoch 3 starts at 2.000 and ends at 3.000, etc.
        
        loss.extend(ep['loss'])
        iou.extend(ep['iou'])

        vloss_eps.append(ep['vloss'])
        viou_eps.append(ep['viou'])

    plot_data(x, loss, iou, savepath='../tmp/diamondback_t-metrics_ep1-20.png')
    plot_data(x_eps, vloss_eps, viou_eps, savepath='../tmp/diamondback_v-metrics_ep1-20.png')