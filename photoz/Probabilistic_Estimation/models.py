from this import d
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfpd = tfp.distributions

class Sampling(tfkl.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class SampleMixtureComponent(tfkl.Layer):
    def call(self, inputs):
        z_mean, z_sig, logits = tf.split(inputs, 3, axis=1)
        idx = tf.cast(tfpd.OneHotCategorical(logits=logits).sample(), tf.bool)
        z_mean_s = tf.expand_dims(tf.boolean_mask(z_mean, idx),axis=1)
        z_sig_s = tf.expand_dims(tf.boolean_mask(z_sig, idx), axis=1)

        return z_mean_s, z_sig_s

class InputColours(tfkl.Layer):
    def __init__(self, n_bands) -> None:
        super().__init__()
        self.n_bands = tf.Variable(n_bands, trainable=False, name='n_bands', dtype=tf.int32)
        self.n_bands_val = self.n_bands.numpy()
    
    def call(self, input):
        diff = tf.experimental.numpy.diff(input)
        diff = tf.concat((diff[:, :self.n_bands_val-1], diff[:, self.n_bands_val:2*self.n_bands_val-1]), -1)
        input_w_colours = tf.concat((input[:, :2*self.n_bands_val+1], diff), -1)

        return input_w_colours

class FreeVariables(tfkl.Layer):
    def __init__(self, units, name):
        super().__init__()

        self.w = self.add_weight(
            shape=(units,), initializer="random_normal",
            trainable=True, name=name
        )

    def call(self, inputs):
        return self.w

class VAE(tfk.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.total_loss_tracker = tfk.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tfk.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tfk.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        data, _ = data
        with tf.GradientTape() as tape:
            (x_mean, x_log_var), (xs, z_mean, z_log_var, z) = self(data)

            log_likelihood = -0.5 * (
                (xs - x_mean) ** 2 / tf.exp(x_log_var)
            ) - 0.5 * tf.math.log(2 * np.pi * tf.exp(x_log_var))

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    log_likelihood,
                    axis=1,  # over dimensions, not the datapoints, [batch_size,15]
                )
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = -(reconstruction_loss - kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        data, _ = data
        (x_mean, x_log_var), (xs, z_mean, z_log_var, z) = self(data)

        log_likelihood = -0.5 * (
            (xs - x_mean) ** 2 / tf.exp(x_log_var)
        ) - 0.5 * tf.math.log(2 * np.pi * tf.exp(x_log_var))

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                log_likelihood,
                axis=1,  # over dimensions, not the datapoints, [batch_size,15]
            )
        )

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        total_loss = -(reconstruction_loss - kl_loss)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function
    def mc_sample(self, input, n_samples = 1, seed=None):
        means, sigmas = tf.split(input, 2, axis=-1)
        batch = tf.shape(means)[0] * n_samples
        dim = tf.shape(means)[1]
        epsilon = tf.cast(
            tf.keras.backend.random_normal(shape=(batch, dim), seed=seed),
            dtype=sigmas.dtype
        )
        return tf.tile(means, [n_samples, 1]) + tf.tile(sigmas,[n_samples,  1]) * epsilon

    def call(self, input, n_samples = 1, seed=None):
        input = self.mc_sample(input, n_samples, seed)
        xs, z_mean, z_log_var, z = self.encoder(input)
        x_mean, x_log_var = self.decoder(z)
        
        return [x_mean, x_log_var], [xs, z_mean, z_log_var, z]

class MDN(tfk.Model):
    def __init__(self, vae, mdn, **kwargs):
        super(MDN, self).__init__(**kwargs)
        self.vae = vae
        self.mdn = mdn
        self.n_mc = tf.Variable(int(self.vae.encoder.layers[1].variables[0].numpy()), name='n_mc', trainable=False, dtype=tf.int32)
        self.mdn_components = tf.Variable(int(self.mdn.output_shape[1]/3), name='mdn_components', trainable=False, dtype=tf.int32)
    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, input, n_samples=1, seed=None):
        [_, _], [_, z_mean, _, _] = self.vae(input, n_samples, seed)
        mdn_pars = self.mdn(z_mean)

        return mdn_pars

#TODO: Rename class atributes etc
class SSGaussRegressorVAE(tfk.Model):
    def __init__(self, encoder, classifier, decoder, alpha = 0.1 , **kwargs):
        super(SSGaussRegressorVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.classifier = classifier
        self.decoder = decoder
        self.alpha = alpha
        self.total_loss_tracker = tfk.metrics.Mean(name="total_loss")
        self.label_reconstruction_loss_tracker = tfk.metrics.Mean(name="label_reconstruction_loss")
        self.labelled_loss_tracker = tfk.metrics.Mean(name="labelled_loss")
        self.unlabelled_loss_tracker = tfk.metrics.Mean(name="unlabelled_loss")
        self.sample_labelled_loss_tracker = tfk.metrics.Mean(name="sample_labelled_loss")
        self.sample_entropy_tracker = tfk.metrics.Mean(name="sample_entropy")
        self.input_reconstruction_loss_tracker = tfk.metrics.Mean(name="input_reconstruction_loss")
        self.latent_space_kl_tracker = tfk.metrics.Mean(name="latent_space_kl")
        self.log_label_prior_tracker = tfk.metrics.Mean(name="label_prior")
        self.rmse_tracker = tfk.metrics.RootMeanSquaredError(name='rmse')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.label_reconstruction_loss_tracker,
            self.labelled_loss_tracker,
            self.unlabelled_loss_tracker,
            self.sample_labelled_loss_tracker,
            self.sample_entropy_tracker,
            self.input_reconstruction_loss_tracker,
            self.latent_space_kl_tracker,
            self.log_label_prior_tracker,
            self.rmse_tracker
        ]
    
    @tf.function
    def mc_sample(self, input, n_samples = 1, seed=None):
        means, sigmas = tf.split(input, 2, axis=-1)
        batch = tf.shape(means)[0] * n_samples
        dim = tf.shape(means)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=seed)
        return tf.tile(means, [n_samples, 1]) + tf.tile(sigmas,[n_samples,  1]) * epsilon
    
    @tf.function
    def input_reconstruction_loss(self, x, x_mean, x_log_var):

        log_likelihood = -0.5 * (
            (x - x_mean) ** 2 / tf.exp(x_log_var)
        ) - 0.5 * tf.math.log(2 * np.pi * tf.exp(x_log_var))

        reconstruction_loss = tf.reduce_sum(
            log_likelihood,
            axis=1,  # over dimensions, not the datapoints
        )

        return reconstruction_loss

    @tf.function
    def gaussian_kl(self, z_mean, z_log_var):

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=1)

        return kl_loss
    
    @tf.function
    def labelled_loss(
        self,
        z_mean, z_log_var,
        x, x_mean, x_log_var,
        labels
    ):
        input_reconstruction_loss = self.input_reconstruction_loss(x, x_mean, x_log_var)
        latent_space_kl = self.gaussian_kl(z_mean, z_log_var)
        log_label_prior = tf.reduce_sum(-0.5 * (labels ** 2) - 0.5 * tf.math.log(2 * np.pi),axis=1)

        labelled_loss = input_reconstruction_loss + log_label_prior - latent_space_kl

        return (labelled_loss, input_reconstruction_loss, latent_space_kl, log_label_prior)
    
    @tf.function
    def unlabelled_loss(
        self,
        z, z_mean, z_log_var,
        x, x_mean, x_log_var,
        r, r_mean, r_log_var
    ):

        sample_labelled_loss = self.labelled_loss(
            z_mean, z_log_var,
            x, x_mean, x_log_var,
            r
        )[0]
        
        sample_entropy = tf.reduce_sum( - (
            -0.5 * ((r - r_mean) ** 2 / tf.exp(r_log_var)) 
            -0.5 * tf.math.log(2 * np.pi * tf.exp(r_log_var))
        ), axis=1)

        unlabelled_loss = sample_labelled_loss + sample_entropy

        return unlabelled_loss, sample_labelled_loss, sample_entropy
    
    @tf.function
    def label_reconstruction_loss(
        self, true_labels, r_mean, r_log_var
    ):

        log_probs = -0.5 * (
            (true_labels - r_mean) ** 2 / tf.exp(r_log_var)
        ) - 0.5 * tf.math.log(2 * np.pi * tf.exp(r_log_var))
        label_reconstruction_loss = - tf.reduce_mean(log_probs)

        return label_reconstruction_loss

    @tf.function
    def conditional_loss_calculation(self, data_labelled, data_unlabelled, true_labels, seed=None):

        # labelled data
        (x_mean_labelled, x_log_var_labelled), (xs_labelled, z_mean_labelled, z_log_var_labelled, z_labelled), (r_mean_labelled, r_log_var_labelled, r_labelled) = self(data_labelled, seed=seed)

        # unlabelled data
        (x_mean_unlabelled, x_log_var_unlabelled), (xs_unlabelled, z_mean_unlabelled, z_log_var_unlabelled, z_unlabelled), (r_mean_unlabelled, r_log_var_unlabelled, r_unlabelled) = self(data_unlabelled, seed=seed)

        labelled_present = tf.math.greater(tf.size(true_labels),0)
        unlabelled_present = tf.math.greater(tf.size(data_unlabelled),0)

        (
            unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss, input_reconstruction_loss,
            latent_space_kl, log_label_prior, label_reconstruction_loss, total_loss, concat_loss
        ) = (
            tf.constant([]), tf.constant([]), tf.constant([]), tf.constant([]), tf.constant([]),
            tf.constant([]), tf.constant([]), tf.constant([]), tf.constant([]), tf.constant([])
        )

        if labelled_present:
            print('Labelled present')

            labelled_loss, input_reconstruction_loss, latent_space_kl, log_label_prior = self.labelled_loss(
                z_mean_labelled, z_log_var_labelled,
                xs_labelled, x_mean_labelled, x_log_var_labelled,
                true_labels
            )

            label_reconstruction_loss = self.label_reconstruction_loss(
                true_labels, r_mean_labelled, r_log_var_labelled
            )

            concat_loss = labelled_loss
            total_loss = (
                - tf.reduce_mean(concat_loss) + self.alpha*label_reconstruction_loss
            )

        if unlabelled_present:
            print('Only unlabelled')
            unlabelled_loss, sample_labelled_loss, sample_entropy = self.unlabelled_loss(
                z_unlabelled, z_mean_unlabelled, z_log_var_unlabelled,
                xs_unlabelled, x_mean_unlabelled, x_log_var_unlabelled,
                r_unlabelled, r_mean_unlabelled, r_log_var_unlabelled
            )

            if labelled_present:
                concat_loss = tf.concat((concat_loss, unlabelled_loss), axis=0)
                total_loss = total_loss = (
                    - tf.reduce_mean(concat_loss) + self.alpha*label_reconstruction_loss
                )
            else:
                concat_loss = unlabelled_loss
                total_loss = -tf.reduce_mean(concat_loss)

        return (
            unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss, input_reconstruction_loss,    
            latent_space_kl, log_label_prior, label_reconstruction_loss, total_loss, r_mean_labelled
        )

    @tf.function
    def conditional_metric_update(
        self, total_loss, unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss,
        input_reconstruction_loss, latent_space_kl, log_label_prior, label_reconstruction_loss, r_mean_labelled, true_labels
    ):

        labelled_present = tf.greater(tf.size(labelled_loss), 0)
        unlabelled_present = tf.greater(tf.size(unlabelled_loss), 0)

        self.total_loss_tracker.update_state(total_loss)

        if labelled_present:
            self.label_reconstruction_loss_tracker.update_state(label_reconstruction_loss)
            self.labelled_loss_tracker.update_state(-tf.reduce_mean(labelled_loss))
            self.input_reconstruction_loss_tracker.update_state(-tf.reduce_mean(input_reconstruction_loss))
            self.latent_space_kl_tracker.update_state(-tf.reduce_mean(latent_space_kl))
            self.log_label_prior_tracker.update_state(-tf.reduce_mean(log_label_prior))
            self.rmse_tracker.update_state(true_labels, r_mean_labelled)
        
        if unlabelled_present:
            self.unlabelled_loss_tracker.update_state(-tf.reduce_mean(unlabelled_loss))
            self.sample_labelled_loss_tracker.update_state(-tf.reduce_mean(sample_labelled_loss))
            self.sample_entropy_tracker.update_state(-tf.reduce_mean(sample_entropy))

        
        metric_states = {
            "loss": self.total_loss_tracker.result(),
            "label_reconstruction_loss": self.label_reconstruction_loss_tracker.result(),
            "labelled_loss": self.labelled_loss_tracker.result(),
            "unlabelled_loss": self.unlabelled_loss_tracker.result(),
            "sample_labelled_loss": self.sample_labelled_loss_tracker.result(),
            "sample_entropy": self.sample_entropy_tracker.result(),
            "input_reconstruction_loss": self.input_reconstruction_loss_tracker.result(),
            "latent_space_kl": self.latent_space_kl_tracker.result(),
            "label_prior": self.log_label_prior_tracker.result(),
            "rmse": self.rmse_tracker.result()
        }

        return metric_states

    def call(self, data, n_samples=1, seed=None):
        xs = self.mc_sample(data, n_samples, seed)
        r_mean, r_log_var, r = self.classifier(xs)
        xs, z_mean, z_log_var, z = self.encoder([xs, r])
        joint_samples = tf.concat((
                z, r
            ), axis=1)
        x_mean, x_log_var = self.decoder(joint_samples)

        return (x_mean, x_log_var), (xs, z_mean, z_log_var, z), (r_mean, r_log_var, r)
        
    def train_step(self, data):
        X, _ = data
        idx_labelled = X[:,-1] != -9999
        data_labelled = tf.boolean_mask(X[:, :-1], idx_labelled)
        data_unlabelled = tf.boolean_mask(X[:, :-1], ~idx_labelled)
        true_labels = tf.expand_dims(tf.boolean_mask(X[:,-1], idx_labelled), axis=1)

        with tf.GradientTape() as tape:

            (
                unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss, input_reconstruction_loss,
                latent_space_kl, log_label_prior, label_reconstruction_loss, total_loss, r_mean_labelled
            ) = self.conditional_loss_calculation(data_labelled, data_unlabelled, true_labels)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        metrics = self.conditional_metric_update(
            total_loss, unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss,
            input_reconstruction_loss, latent_space_kl, log_label_prior, label_reconstruction_loss, r_mean_labelled, true_labels
        )

        return metrics
    
    def test_step(self, data, seed=None):
        X, _ = data
        idx_labelled = X[:,-1] != -9999
        data_labelled = tf.boolean_mask(X[:, :-1], idx_labelled)
        data_unlabelled = tf.boolean_mask(X[:, :-1], ~idx_labelled)
        true_labels = tf.expand_dims(tf.boolean_mask(X[:,-1], idx_labelled), axis=1)

        (
            unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss, input_reconstruction_loss,
            latent_space_kl, log_label_prior, label_reconstruction_loss, total_loss, r_mean_labelled
        ) = self.conditional_loss_calculation(data_labelled, data_unlabelled, true_labels, seed=seed)

        metrics = self.conditional_metric_update(
            total_loss, unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss,
            input_reconstruction_loss, latent_space_kl, log_label_prior, label_reconstruction_loss, r_mean_labelled, true_labels
        )

        return metrics

#TODO: Rename class atributes etc
class SSMDNRegressorVAE(tfk.Model):
    def __init__(self, encoder, classifier, decoder, alpha = 0.1 , **kwargs):
        super(SSMDNRegressorVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.classifier = classifier
        self.decoder = decoder
        self.alpha = alpha
        self.total_loss_tracker = tfk.metrics.Mean(name="total_loss")
        self.label_reconstruction_loss_tracker = tfk.metrics.Mean(name="label_reconstruction_loss")
        self.labelled_loss_tracker = tfk.metrics.Mean(name="labelled_loss")
        self.unlabelled_loss_tracker = tfk.metrics.Mean(name="unlabelled_loss")
        self.sample_labelled_loss_tracker = tfk.metrics.Mean(name="sample_labelled_loss")
        self.sample_entropy_tracker = tfk.metrics.Mean(name="sample_entropy")
        self.input_reconstruction_loss_tracker = tfk.metrics.Mean(name="input_reconstruction_loss")
        self.latent_space_kl_tracker = tfk.metrics.Mean(name="latent_space_kl")
        self.log_label_prior_tracker = tfk.metrics.Mean(name="label_prior")
        self.rmse_tracker = tfk.metrics.RootMeanSquaredError(name='rmse')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.label_reconstruction_loss_tracker,
            self.labelled_loss_tracker,
            self.unlabelled_loss_tracker,
            self.sample_labelled_loss_tracker,
            self.sample_entropy_tracker,
            self.input_reconstruction_loss_tracker,
            self.latent_space_kl_tracker,
            self.log_label_prior_tracker,
            self.rmse_tracker
        ]
    
    @tf.function
    def mc_sample(self, input, n_samples = 1, seed=None):
        means, sigmas = tf.split(input, 2, axis=-1)
        batch = tf.shape(means)[0] * n_samples
        dim = tf.shape(means)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=None)
        return tf.tile(means, [n_samples, 1]) + tf.tile(sigmas,[n_samples,  1]) * epsilon
    
    @tf.function
    def input_reconstruction_loss(self, x, x_mean, x_log_var):

        log_likelihood = -0.5 * (
            (x - x_mean) ** 2 / tf.exp(x_log_var)
        ) - 0.5 * tf.math.log(2 * np.pi * tf.exp(x_log_var))

        reconstruction_loss = tf.reduce_sum(
            log_likelihood,
            axis=1,  # over dimensions, not the datapoints
        )

        return reconstruction_loss

    @tf.function
    def gaussian_kl(self, z_mean, z_log_var):

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=1)

        return kl_loss
    
    @tf.function
    def labelled_loss(
        self,
        z_mean, z_log_var,
        x, x_mean, x_log_var,
        labels
    ):
        input_reconstruction_loss = self.input_reconstruction_loss(x, x_mean, x_log_var)
        latent_space_kl = self.gaussian_kl(z_mean, z_log_var)
        log_label_prior = tf.reduce_sum(-0.5 * (labels ** 2) - 0.5 * tf.math.log(2 * np.pi),axis=1)

        labelled_loss = input_reconstruction_loss + log_label_prior - latent_space_kl

        return (labelled_loss, input_reconstruction_loss, latent_space_kl, log_label_prior)
    
    @tf.function
    def unlabelled_loss(
        self,
        z, z_mean, z_log_var,
        x, x_mean, x_log_var,
        r, r_mdn_pars
    ):

        sample_labelled_loss = self.labelled_loss(
            z_mean, z_log_var,
            x, x_mean, x_log_var,
            r
        )[0]
        
        mus, sigs, alpha_logits = tf.split(r_mdn_pars, 3, axis=-1)
        alphas = tf.nn.softmax(alpha_logits, axis=1)
        sample_entropy = -tf.math.reduce_logsumexp(
            tf.math.log(alphas) - 0.5 * tf.math.log(2 * np.pi) - tf.math.log(sigs)
            - 0.5 * (r - mus) ** 2 / sigs ** 2, axis=1
        )

        unlabelled_loss = sample_labelled_loss + sample_entropy

        return unlabelled_loss, sample_labelled_loss, sample_entropy
    
    @tf.function
    def label_reconstruction_loss(
        self, true_labels, r_mdn_pars
    ):

        mus, sigs, alpha_logits = tf.split(r_mdn_pars, 3, axis=-1)
        alphas = tf.nn.softmax(alpha_logits, axis=1)
        log_probs = tf.math.reduce_logsumexp(
            tf.math.log(alphas) - 0.5 * tf.math.log(2 * np.pi) - tf.math.log(sigs)
            - 0.5 * (true_labels - mus) ** 2 / sigs ** 2, axis=1
        )
        label_reconstruction_loss = - tf.reduce_mean(log_probs)

        return label_reconstruction_loss

    @tf.function
    def conditional_loss_calculation(self, data_labelled, data_unlabelled, true_labels):

        # labelled data
        (x_mean_labelled, x_log_var_labelled), (xs_labelled, z_mean_labelled, z_log_var_labelled, z_labelled), (r_mdn_pars_labelled, r_labelled) = self(data_labelled)

        # unlabelled data
        (x_mean_unlabelled, x_log_var_unlabelled), (xs_unlabelled, z_mean_unlabelled, z_log_var_unlabelled, z_unlabelled), (r_mdn_pars_unlabelled, r_unlabelled) = self(data_unlabelled)

        labelled_present = tf.math.greater(tf.size(true_labels),0)
        unlabelled_present = tf.math.greater(tf.size(data_unlabelled),0)

        (
            unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss, input_reconstruction_loss,
            latent_space_kl, log_label_prior, label_reconstruction_loss, concat_loss
        ) = (
            tf.constant([]), tf.constant([]), tf.constant([]), tf.constant([]), tf.constant([]),
            tf.constant([]), tf.constant([]), tf.constant([]), tf.constant([])
        )

        if labelled_present:
            print('Labelled present')

            labelled_loss, input_reconstruction_loss, latent_space_kl, log_label_prior = self.labelled_loss(
                z_mean_labelled, z_log_var_labelled,
                xs_labelled, x_mean_labelled, x_log_var_labelled,
                true_labels
            )

            label_reconstruction_loss = self.label_reconstruction_loss(
                true_labels, r_mdn_pars_labelled
            )

            concat_loss = labelled_loss

        if unlabelled_present:
            print('Only unlabelled')
            unlabelled_loss, sample_labelled_loss, sample_entropy = self.unlabelled_loss(
                z_unlabelled, z_mean_unlabelled, z_log_var_unlabelled,
                xs_unlabelled, x_mean_unlabelled, x_log_var_unlabelled,
                r_unlabelled, r_mdn_pars_unlabelled
            )

            if labelled_present:
                concat_loss = tf.concat((concat_loss, unlabelled_loss), axis=0)
            else:
                concat_loss = unlabelled_loss

        return (
            unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss, input_reconstruction_loss,    
            latent_space_kl, log_label_prior, label_reconstruction_loss, concat_loss, r_mdn_pars_labelled
        )

    @tf.function
    def conditional_metric_update(
        self, total_loss, unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss,
        input_reconstruction_loss, latent_space_kl, log_label_prior, label_reconstruction_loss, r_mdn_pars_labelled, true_labels
    ):

        labelled_present = tf.greater(tf.size(labelled_loss), 0)
        unlabelled_present = tf.greater(tf.size(unlabelled_loss), 0)

        self.total_loss_tracker.update_state(total_loss)

        if labelled_present:
            mus, _, alpha_logits = tf.split(r_mdn_pars_labelled, 3, axis=1)
            alpha = tf.nn.softmax(alpha_logits, axis=1)
            mdn_mus = tf.reduce_sum(mus * alpha, axis=1)
            self.label_reconstruction_loss_tracker.update_state(label_reconstruction_loss)
            self.labelled_loss_tracker.update_state(-tf.reduce_mean(labelled_loss))
            self.input_reconstruction_loss_tracker.update_state(-tf.reduce_mean(input_reconstruction_loss))
            self.latent_space_kl_tracker.update_state(-tf.reduce_mean(latent_space_kl))
            self.log_label_prior_tracker.update_state(-tf.reduce_mean(log_label_prior))
            self.rmse_tracker.update_state(true_labels, mdn_mus)
        
        if unlabelled_present:
            self.unlabelled_loss_tracker.update_state(-tf.reduce_mean(unlabelled_loss))
            self.sample_labelled_loss_tracker.update_state(-tf.reduce_mean(sample_labelled_loss))
            self.sample_entropy_tracker.update_state(-tf.reduce_mean(sample_entropy))

        
        metric_states = {
            "loss": self.total_loss_tracker.result(),
            "label_reconstruction_loss": self.label_reconstruction_loss_tracker.result(),
            "labelled_loss": self.labelled_loss_tracker.result(),
            "unlabelled_loss": self.unlabelled_loss_tracker.result(),
            "sample_labelled_loss": self.sample_labelled_loss_tracker.result(),
            "sample_entropy": self.sample_entropy_tracker.result(),
            "input_reconstruction_loss": self.input_reconstruction_loss_tracker.result(),
            "latent_space_kl": self.latent_space_kl_tracker.result(),
            "label_prior": self.log_label_prior_tracker.result(),
            "rmse": self.rmse_tracker.result()
        }

        return metric_states

    def call(self, data, n_samples=1, seed=None):
        xs = self.mc_sample(data, n_samples, seed)
        r_mdn_pars, r = self.classifier(xs)
        # assert_op = tf.debugging.Assert(
        #     tf.equal(tf.cast(tf.size(r),tf.int32), tf.cast(tf.size(xs) / 15, tf.int32)),
        #     [tf.size(data), tf.size(xs), tf.size(r), xs, r_mdn_pars, r],
        #     summarize=100
        # )
        xs, z_mean, z_log_var, z = self.encoder([xs, r])
        joint_samples = tf.concat((
                z, r
            ), axis=1)
        x_mean, x_log_var = self.decoder(joint_samples)

        return (x_mean, x_log_var), (xs, z_mean, z_log_var, z), (r_mdn_pars, r)
        
    def train_step(self, data):
        X, _ = data
        idx_labelled = X[:,-1] != -9999
        data_labelled = tf.boolean_mask(X[:, :-1], idx_labelled)
        data_unlabelled = tf.boolean_mask(X[:, :-1], ~idx_labelled)
        true_labels = tf.expand_dims(tf.boolean_mask(X[:,-1], idx_labelled), axis=1)

        with tf.GradientTape() as tape:

            (
                unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss, input_reconstruction_loss,
                latent_space_kl, log_label_prior, label_reconstruction_loss, concat_loss, r_mdn_pars_labelled
            ) = self.conditional_loss_calculation(data_labelled, data_unlabelled, true_labels)
            
            total_loss = (
                - tf.reduce_mean(concat_loss) + self.alpha*label_reconstruction_loss
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        metrics = self.conditional_metric_update(
            total_loss, unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss,
            input_reconstruction_loss, latent_space_kl, log_label_prior, label_reconstruction_loss, r_mdn_pars_labelled, true_labels
        )

        return metrics
    
    def test_step(self, data):
        X, _ = data
        idx_labelled = X[:,-1] != -9999
        data_labelled = tf.boolean_mask(X[:, :-1], idx_labelled)
        data_unlabelled = tf.boolean_mask(X[:, :-1], ~idx_labelled)
        true_labels = tf.expand_dims(tf.boolean_mask(X[:,-1], idx_labelled), axis=1)

        (
            unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss, input_reconstruction_loss,
            latent_space_kl, log_label_prior, label_reconstruction_loss, concat_loss, r_mdn_pars_labelled
        ) = self.conditional_loss_calculation(data_labelled, data_unlabelled, true_labels)
        
        total_loss = (
            - tf.reduce_mean(concat_loss) + self.alpha*label_reconstruction_loss
        )

        metrics = self.conditional_metric_update(
            total_loss, unlabelled_loss, sample_labelled_loss, sample_entropy, labelled_loss,
            input_reconstruction_loss, latent_space_kl, log_label_prior, label_reconstruction_loss, r_mdn_pars_labelled, true_labels
        )

        return metrics
