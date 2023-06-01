
import library.quaternionUtils as qu
import matplotlib.pyplot as plt

def display_quat_as_euler(
    q_arrays: list, 
    t_arrays: list, 
    labels: list = None, 
    title: str = '',
    seq: str = 'XYZ',
    degrees: bool = True
    ):

    # Convert to euler representation 
    for i, q_arr in enumerate(q_arrays):
        q_arrays[i] = qu.quaternionToEuler(q_arr, seq, degrees)

    # Plot section
    sub_title = {'X': 'Pitch', 'Y': 'Roll', 'Z': 'Yaw'}
    TITLES = [sub_title[s] for s in seq]
    plt.figure(figsize = (10,9))
    plt.tight_layout(pad = 5)
    for i in range(3):
        plt.subplot(2, 2, i+1)
        for q_arr, t_arr, l in zip(q_arrays, t_arrays, labels):
            plt.plot(t_arr, q_arr[:,i], label=l) 
        plt.ylabel(r'Wartość w stopniach $(^{\circ})$')
        plt.xlabel('N-ta próbka')
        plt.title(TITLES[i])
        plt.legend(loc='upper right')

    plt.suptitle(title)
    plt.show()