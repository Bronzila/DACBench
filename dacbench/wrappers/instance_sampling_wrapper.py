from gym import Wrapper
import numpy as np
from scipy.stats import norm


class InstanceSamplingWrapper(Wrapper):
    def __init__(self, env, sampling_function=None, instances=None):
        super(InstanceSamplingWrapper, self).__init__(env)
        if sampling_function:
            self.sampling_function = sampling_function
        elif instances:
            self.sampling_function = self.fit_dist(instances)
        else:
            raise Exception("No distribution to sample from given")

    def __setattr__(self, name, value):
        if name in ["sampling_function", "env", "fit_dist", "reset"]:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def __getattribute__(self, name):
        if name in ["sampling_function", "env", "fit_dist", "reset"]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self.env, name)

    def reset(self):
        """
        Reset environment and use sampled instance for training

        Returns
        -------
        np.array
            state
        """
        instance = self.sampling_function()
        self.env.set_instance_set([instance])
        self.env.set_inst_id(0)
        return self.env.reset()

    def fit_dist(self, instances):
        """
        Approximate instance distribution in given instance set

        Parameters
        ----------
        instances : List
            instance set

        Returns
        ---------
        method
            sampling method for new instances
        """
        dists = []
        for i in range(len(instances[0])):
            component = [inst[i] for inst in instances]
            dist = norm.fit(component)
            dists.append(dist)

        def sample():
            instance = []
            for d in dists:
                instance.append(np.random.normal(d[0], d[1]))
            return instance

        return sample