import numpy as np
from matplotlib import pyplot as plt


"""
These numbers were (painfully) extracted and formatted from the keras logs
under logs/. 

Epochs 1-7 are in one file, after which we added l2 reg, and then
did epochs 8-20.

UPDATE: We are only plotting metrics over epochs.
"""

epoch1 = {
    'loss' : [378364.0312, 119044.3281, 80540.8672, 72485.3631, 68270.9631, 65601.2418, 63880.2223, 62391.8667, 61146.4818, 60260.3811],
    'iou' : [0.1665, 0.4204, 0.5703, 0.6061, 0.6268, 0.6396, 0.6475, 0.6538, 0.6596, 0.6636],
    'vloss' : 61352.2011,
    'viou' : 0.6909
}

epoch2 = {
    'loss' : [61078.7598, 52234.4410, 51989.6470, 51951.3594, 51668.4240, 51090.3679, 50954.0791, 50795.9881, 50681.1486],
    'iou' : [0.6429, 0.7076, 0.7045, 0.7046, 0.7053, 0.7071, 0.7071, 0.7074, 0.7087],
    'vloss' : 52981.4480,
    'viou' : 0.7031
}

epoch3 = {
    'loss' : [28614.7698, 48403.9542, 48540.6464, 48439.8972, 48378.9897, 48316.0248, 48048.8650, 47823.2336, 47693.9662],
    'iou' : [0.6233, 0.7195, 0.7211, 0.7206, 0.7203, 0.7209, 0.7220, 0.7224, 0.7230],
    'vloss' : 52984.8885,
    'viou' : 0.7056
}

epoch4 = {
    'loss' : [55985.7891, 46152.6722, 45841.4347, 46056.1062, 45911.6120, 45874.9179, 45910.3057, 45929.5863, 45995.2859],
    'iou' : [0.7498, 0.7274, 0.7283, 0.7287, 0.7304, 0.7305, 0.7319, 0.7318, 0.7318],
    'vloss' : 50369.4005,
    'viou' : 0.7146
}

epoch5 = {
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
    'loss' : [38539.5698, 42349.6251, 42251.4041, 42428.7473, 42233.3160, 42358.8873, 42499.2086, 42466.9905, 42375.9954],
    'iou' : [0.7671, 0.7528, 0.7494, 0.7487, 0.7506, 0.7500, 0.7497, 0.7499, 0.7505],
    'vloss' : 47624.5262,
    'viou' : 0.7310
}

epoch8 = {
    'loss' : 41555.6282,
    'iou' : 0.7552,
    'vloss' : 53946.5621,
    'viou' : 0.7258
}

epoch9 = {
    'loss' : 40294.3173,
    'iou' : 0.7606,
    'vloss' : 47224.1838,
    'viou' : 0.7336
}

epoch10 = {
    'loss' : 39362.0871,
    'iou' : 0.7655,
    'vloss' : 47845.0906,
    'viou' : 0.7321
}

epoch11 = {
    'loss' : 38692.3023,
    'iou' : 0.7693,
    'vloss' : 49392.7360,
    'viou' : 0.7317
}

epoch12 = {
    'loss' : 37958.9998,
    'iou' : 0.7734,
    'vloss' : 47164.9414,
    'viou' : 0.7436
}

epoch13 = {
    'loss' : 37463.3230,
    'iou' : 0.7756,
    'vloss' : 47330.7253,
    'viou' : 0.7472
}

epoch14 = {
    'loss' : 34853.0411,
    'iou' : 0.7893,
    'vloss' : 44811.2473,
    'viou' : 0.7522
}

epoch15 = {
    'loss' : 35438.9244,
    'iou' : 0.7864,
    'vloss' : 45091.2064,
    'viou' : 0.7472
}

epoch16 = {
    'loss' : 35037.9977,
    'iou' : 0.7886,
    'vloss' : 45078.5233,
    'viou' : 0.7482
}

epoch17 = {
    'loss' : 34734.5441,
    'iou' : 0.7897,
    'vloss' : 46438.9899,
    'viou' : 0.7498
}

epoch18 = {
    'loss' : 34421.9269,
    'iou' : 0.7913,
    'vloss' : 44111.4496,
    'viou' : 0.7554
}

epoch19 = {
    'loss' : 34142.9699,
    'iou' : 0.7932,
    'vloss' : 46381.3267,
    'viou' : 0.7475
}


def printl(obj, name):
    print("%s: %s" % (name, str(obj)))


def plot_data_old(x, loss, iou, savepath=None):
    """ If no savepath, save in curr dir.
    """

    # Training
    fig, ax1 = plt.subplots()
    
    ax1.plot(x, loss, 'b-')
    ax1.set_xlabel('Number of completed epochs')
    ax1.set_ylabel('Total cross entropy loss', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xticks(range(1, 20, 1))

    ax2 = ax1.twinx()
    ax2.plot(x, iou, 'g-')
    ax2.set_ylabel('IOU', color='g')
    ax2.tick_params('y', colors='g')
    ax1.set_xticks(range(1, 20, 1))

    fig.tight_layout()

    ax1.grid(True)

    if savepath is not None:
        plt.savefig(savepath)
    else:
        from datetime import datetime
        plt.savefig("figure_%s.png" % str(datetime.now()))

def plot_data(x, traindata, valdata, ylabel, loc, savepath=None):

    line_t, = plt.plot(x, traindata, '.-', color="#F97306", label='train')
    line_v, = plt.plot(x, valdata, '.-', color="#00FFFF", label='val')
    plt.xlabel('Number of completed epochs')
    plt.ylabel(ylabel)
    plt.xticks(range(1, 20, 1))

    plt.grid(True)
    plt.legend(handles=[line_t, line_v], loc=loc)

    if savepath is not None:
        plt.savefig(savepath)
    else:
        from datetime import datetime
        plt.savefig("figure_%s.png" % str(datetime.now()))

    plt.close()

if __name__ == "__main__":
    local_vars = dict(locals()) # shallow copy to avoid concurrency problems
    epochs = [(int(v[5:]), local_vars[v]) for v in local_vars if 'epoch' in v]
    epochs = sorted(epochs, key=lambda tup: tup[0])
    epochs = [tup[1] for tup in epochs]

    x = list(range(1, len(epochs)+1))
    loss = []
    iou = []
    vloss = []
    viou = []
    
    for ep in epochs:
        # Back compatibility with our list of train metrics
        if isinstance(ep['loss'], list):
            loss.append(ep['loss'][-1])
            iou.append(ep['iou'][-1])
        else:
            loss.append(ep['loss'])
            iou.append(ep['iou'])

        vloss.append(ep['vloss'])
        viou.append(ep['viou'])
    
    ylabel_loss = 'Total cross entropy loss'
    ylabel_iou = "IOU"

    plot_data(x, iou, viou, ylabel_iou, 'lower right', savepath='assets/diamondback_t-metrics_ep1-20.png')
    plot_data(x, loss, vloss, ylabel_loss, 'upper right', savepath='assets/diamondback_v-metrics_ep1-20.png')