import roc
from confusion_matrices import bernoulli_cm2_binary
from FnB_LSTM import FnB_LSTMtan_N, FnB_LSTMsig_N
from oned_convlayer_rectify import *
from custom_layers import High_divx
from custom_layers import High_divabs
from custom_layers import Sum_ax1
from custom_layers import Sum_last_ax
from custom_layers import Semi_soft
from custom_layers import Divide_to_one

import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys
import time


class Network:
    """
        This is the CNN-BLSTM class. Contains functions common to all CNN-BLSTM networks.
    """

    def __init__(self, **kwargs):
        # assign default values that must be present, or else the network will not work
        self.options = {
            "networktype": "CNN-BLSTM",
            "NUMBER_OF_CLASSES": 1,
            "N_L1": 200,
            "N_L2": 200,
            "DROPOUT_IN": 0.,
            "DROPOUT_LSTM": 0.1,
            "DROPOUT_OUT": 0.5,
            "DENSELAYER_NODES": 100,
	    "L2": 0.00,
            "early_stopping": 10,

        }

        # load user supplied options
        for k in kwargs.keys():
            self.options[k] = kwargs[k]

        # define some variables
        self.options["BS_PR_SEQ"] = self.options["SEQ_SIZE"]  # bases per sequence - actual sequence length
        self.options["FS"] = [
            self.options["BS_PR_SEQ"] - (self.options["FILTER_SIZES"][i] / len(self.options["VOCAB"])) + 1 for i in
            range(len(self.options["FILTER_SIZES"]))
            ]
        self.options["ALL_F"] = sum(self.options["FS"])
        self.options["NUMBER_OF_CONV_LAYERS"] = len(self.options["FILTER_SIZES"])

        # temporary compatibility fix
        self.type = self.options["networktype"]
        self.VOCAB = self.options["VOCAB"]
        self.FS = self.options["FS"]
        self.ALL_F = self.options["ALL_F"]
        self.BS_PR_SEQ = self.options["BS_PR_SEQ"]
        #self.DROPOUT_LSTM = self.options["DROPOUT_LSTM"]
        self.GRAD_CLIP = self.options["GRAD_CLIP"]
        self.FILTER_SIZES = self.options["FILTER_SIZES"]

        #######################################################
        # symbolic variables                                  #
        #######################################################
        # Theano defines its computations using symbolic variables. A symbolic variable
        # is a matrix, vector, 3D matrix and specifies the data type.
        # A symbolic value does not hold any data, like a matlab matrix or np.array
        # Note that mask is constructed with a broadcastable argument which specifies
        # that the mask can be broadcasted in the 3. dimension.
        self.sym_input = T.tensor3('inputs')
        self.sym_target = T.icol('targets')

        # finally, build the model layers
        self.build_model()

    def set_seqsize(self, seqsize):
        self.options["SEQ_SIZE"] = seqsize
        return self.options["SEQ_SIZE"]

    def get_seqsize(self):
        return self.options["SEQ_SIZE"]

    def build_model(self):
        network = {}
        # Input layer, as usual:
        network['l_in'] = lasagne.layers.InputLayer(
            shape=(None, 1, len(self.options["VOCAB"]) * self.options["SEQ_SIZE"]),
            input_var=self.sym_input)

        # symbolic variables that we can use later
        self.batchsize, n_features, onehot_length = network['l_in'].input_var.shape

        network['l_in'] = lasagne.layers.DropoutLayer(network['l_in'], p=self.options["DROPOUT_IN"])

        # THE FUNCTION 'setup_conv_layers' CONTAINS THE CONVOLUTIONAL LAYERS
        c_layers, pool_layers, full, full2 = setup_conv_layers(network['l_in'], self.options["FS"],self.batchsize,
                                                        self.options["SEQ_SIZE"],
                                                  filters_per_convolution=self.options["FILTERS"],
                                                  filter_sizes=self.options["FILTER_SIZES"],
                                                  vocab=self.options["VOCAB"],
                                                  dropout_conv=self.options["DROPOUT_CONV"],
                                                  padding=self.options["PADDING"])

        network['l_concat'] = lasagne.layers.ConcatLayer(incomings=c_layers, axis=2)
        #network['l_concat'] = High_divx(network['l_concat'])
        network['l_concat_arg'] = lasagne.layers.ConcatLayer(incomings=pool_layers, axis=2)


        network['cn_layers'] = lasagne.layers.ConcatLayer(incomings=full2, axis=2)
        network['l_sumc'] = lasagne.layers.ConcatLayer(incomings=full, axis=1)
        network['l_sumc'] = lasagne.layers.ReshapeLayer(network['l_sumc'], (
            self.batchsize, len(self.options["FILTER_SIZES"]), len(self.options["VOCAB"]) * self.options["SEQ_SIZE"]))


        network['l_res_sumc'] = lasagne.layers.ReshapeLayer(network['l_sumc'], (
            self.batchsize, len(self.options["FILTER_SIZES"]), len(self.options["VOCAB"]) * self.options["SEQ_SIZE"]))
        network['l_sum_rsumc'] = Sum_ax1(network['l_res_sumc'])
        network['l_res_ssum'] = lasagne.layers.ReshapeLayer(network['l_sum_rsumc'], (
            self.batchsize, 1, len(self.options["VOCAB"]) * self.options["SEQ_SIZE"]))
        network['l_res_ssum'] = High_divx(network['l_res_ssum'])

        #network['l_res_ssum'] = lasagne.layers.ReshapeLayer(network['l_res_ssum'], (
        #    batchsize, self.options["BS_PR_SEQ"], len(self.options["VOCAB"])))

        network['inp'] = lasagne.layers.ReshapeLayer(network['l_in'],
                                                     (self.batchsize, self.options["BS_PR_SEQ"], len(self.options["VOCAB"])))

	network['inp_one'] = Divide_to_one(network['inp'])

        network['l_lstmin'] = lasagne.layers.ConcatLayer(incomings=[network['cn_layers'], network['inp']], axis=2)

        lstm_f, lstm_b = FnB_LSTMtan_N(network['l_lstmin'], network['l_lstmin'], self.options["N_LSTM"],
                                             self.options["DROPOUT_LSTM"], self.GRAD_CLIP, forget_b=0, ing=0., cellg=0., outg=0.)


        # concat them into one final output
        network['l_sumz'] = lasagne.layers.ConcatLayer(incomings=[lstm_f, lstm_b], axis=2)

	#lstm_f1, lstm_b1 = FnB_LSTMsig_N(network['l_sumz'], network['l_sumz'], 2,
        #                                     self.DROPOUT_LSTM, self.GRAD_CLIP, forget_b=0, ing=0., cellg=0., outg=0.)

        #network["resh"] = lasagne.layers.ReshapeLayer( lasagne.layers.dropout(network['l_sumz'], p=self.options["DROPOUT_OUT"]) , (self.batchsize * self.options["BS_PR_SEQ"], 2*self.options["N_LSTM"]))

	#network["resh"] = lasagne.layers.DenseLayer(network["resh"], num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.Constant(0.1), b=None)

	#network['resh'] = lasagne.layers.ConcatLayer(incomings=[lstm_f1, lstm_b1], axis=2)

	#network["resh"] = lasagne.layers.ReshapeLayer( network["resh"] , (self.batchsize , self.options["BS_PR_SEQ"], 1))
	#network['resh'] = Sum_last_ax(network['resh'])


        network['l_sumz2x'] = Sum_last_ax(network['l_sumz'])
        network['l_sumz2x'] = lasagne.layers.ReshapeLayer(network['l_sumz2x'],
                                                          (self.batchsize, self.options["BS_PR_SEQ"], 1))
        network['l_sumz2x'] = lasagne.layers.ConcatLayer(
            [network['l_sumz2x'], network['l_sumz2x'], network['l_sumz2x'], network['l_sumz2x']], axis=2)
        network['l_sumz2x'] = lasagne.layers.ReshapeLayer(network['l_sumz2x'], (
            self.batchsize, 1, len(self.options["VOCAB"]) * self.options["SEQ_SIZE"]))
        network['l_sumz2x'] = lasagne.layers.ElemwiseMergeLayer([network['l_in'], network['l_sumz2x']],
                                                                theano.tensor.mul)

	#network['l_sumz2x'] = lasagne.layers.ReshapeLayer(network['l_sumz2x'], (
        #    self.batchsize, 1,  self.options["SEQ_SIZE"]* len(self.options["VOCAB"]) ))

	network['l_sumz2x'] = lasagne.layers.ReshapeLayer(network['l_sumz2x'], (
            self.batchsize,  self.options["SEQ_SIZE"], len(self.options["VOCAB"]) ))


	network['l_profile'] = lasagne.layers.DropoutLayer(Sum_last_ax(network['l_sumz2x']), p=self.options["DROPOUT_OUT"])
        print " Dropout:",str(self.options["DROPOUT_OUT"])

        #network['l_attention'] = High_divabs(network['l_sumz2x'])
	#network['l_attention'] = network['l_sumz2x']

        network['l_out'] = lasagne.layers.DenseLayer(network['l_profile'], num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid,
                                                     W=lasagne.init.Constant(1.0), b=None)  # 0.001

	#network['l_out'] = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(network['l_sumz'], p=self.options["DROPOUT_OUT"]),
	#					     num_units=1,
        #                                             nonlinearity=lasagne.nonlinearities.sigmoid,
        #                                             W=lasagne.init.Constant(0.1), b=None)  # this layer is the overall classifier of the entire sequence

	network['output_params'] = network['l_out'].get_params()
	network['output_params'] = network['output_params'][0]


        self.network = network

    def compile_training_functions(self):
        # print parameter info
        all_params = lasagne.layers.get_all_params(self.network['l_profile'], trainable=True) #l_out
        total_params = sum([p.get_value().size for p in all_params])
        print " Total Model Parameters:", total_params
        print " Trainable Model Parameters"

	all_params = lasagne.layers.get_all_params(self.network['l_profile'], trainable=True) #l_out
	total_params = sum([p.get_value().size for p in all_params])
        print " Total Model Parameters:", total_params

	print "-" * 40
        for param in all_params:
            print '', param, param.get_value().shape
        print "-" * 40
        sys.stdout.flush()

        # train cost
        train_preds = lasagne.layers.get_output(self.network['l_out'], deterministic=False)
        cost_train = T.mean(lasagne.objectives.binary_crossentropy(train_preds, self.sym_target))
        L1_n_L2 = lasagne.regularization.regularize_network_params(self.network['l_out'],
                                                                   lasagne.regularization.l2,
                                                                   {'regularizable': True})
        cost_train += L1_n_L2 * self.options["L2"]
        eq_train = T.eq(T.round(train_preds), self.sym_target)
        train_acc = T.mean(eq_train, dtype=theano.config.floatX)

        # validation cost and accuracy
        val_preds = lasagne.layers.get_output(self.network['l_out'], deterministic=True)
        cost_val = T.mean(lasagne.objectives.binary_crossentropy(val_preds, self.sym_target))
        eq_val = T.eq(T.round(val_preds), self.sym_target)
        val_acc = T.mean(eq_val, dtype=theano.config.floatX)

        print " Making update function...",
        sys.stdout.flush()
        updates = lasagne.updates.adam(cost_train, all_params, learning_rate=self.options["ETA"], beta1=0.9,
                                       beta2=0.999, epsilon=1e-08)
        print "done"
        print " Making Theano training function - slow step...",
        sys.stdout.flush()
        start_time = time.time()
        self.train_fn = theano.function([self.sym_input, self.sym_target], [cost_train, train_acc, train_preds],
                                        updates=updates, allow_input_downcast=True)
        ctime = time.time() - start_time
        print("finished in {:.3f}s".format(ctime))
        print " Making Theano validation function - slow step...",
        sys.stdout.flush()
        start_time = time.time()
        self.val_fn = theano.function([self.sym_input, self.sym_target], [cost_val, val_acc, val_preds, eq_val],
                                      allow_input_downcast=True)
        ctime = time.time() - start_time
        print("finished in {:.3f}s".format(ctime))
        sys.stdout.flush()

    def compile_prediction_function(self):
        print " Making Theano prediction function ...",
        sys.stdout.flush()
        start_time = time.time()
        prediction, profile, conv_weight, lstm_weight = lasagne.layers.get_output(
            [self.network['l_out'], self.network['l_concat_arg'], self.network['l_concat'],
             self.network['l_profile']],
            deterministic=True)

        prediction_fn= theano.function([self.sym_input],
                               [prediction, profile, conv_weight, lstm_weight],
                               allow_input_downcast=True)
        ctime = time.time() - start_time
        print("finished in {:.3f}s".format(ctime))
        sys.stdout.flush()
        return prediction_fn, self.network['output_params']

    def fit(self, all_inputs, num_epochs=24):


        sys.stdout.flush()

        self.compile_training_functions()

        auc_cv = []
        roc_cv = []
        cv_results = []
        cfx = 0

        if self.options["runmode"] == "cv":
            init_params = lasagne.layers.get_all_param_values(self.network['l_sumz'])
            #print ' Initial parameters have been saved'
            if '_cv_cycle_data.pkl' in self.options['cvfile']:
                print " Loading previous CV cycle data from ", str(self.options['cvfile'])
                with open(self.options['cvfile'], "rb") as output:
                    init_params, auc_cv = pickle.load(output)
                self.options['cvfile'] = self.options['cvfile'].replace('_cv_cycle_data.pkl','')
                lasagne.layers.set_all_param_values(self.network['l_sumz'], init_params)
                cfx = len(auc_cv)
                print ' Starting CV from set ' + str(cfx+1)

        for cf in range(cfx, len(all_inputs)):
            best_val = 0
            best_auroc = 0
            best_loss = 10000000
            best_epoch = 0
            best_epoch_acc = 0
            best_epoch_loss = 0
            breaker = 0
            train_all = []
            test_all = []
            cm_all = []

	    if self.options["runmode"] == "cv":
		#train, val, test, tr_sqs, tr_ids, va_sqs, va_ids, te_sqs, te_ids = all_inputs[cf]
		train, val, test = all_inputs[cf]
            	X_train, y_train = train
            	X_val, y_val = val
            	X_test, y_test = test


	    if self.options["runmode"] != "cv":
                train, val, test = all_inputs[cf]
                X_train, y_train = train
                X_val, y_val = val
                X_test, y_test = test

            if len(all_inputs) > 1:
                print "\n\n Processing CV set {}".format(cf + 1)
            print ' Items in train list: {}'.format(X_train.shape)
            print ' Items in validation list: {}'.format(X_val.shape)
            print ' Items in train-target list: {}'.format(y_train.shape)
            print ' Items in validation-target list: {}'.format(y_val.shape)


            X_train = X_train.reshape((-1, 1, len(self.options["VOCAB"]) * self.options["SEQ_SIZE"]))
            X_val = X_val.reshape((-1, 1, len(self.options["VOCAB"]) * self.options["SEQ_SIZE"]))
            X_test = X_test.reshape((-1, 1, len(self.options["VOCAB"]) * self.options["SEQ_SIZE"]))


            if len(all_inputs) == 1:
                cf = ""

            early_stopping = 0
            #previous_auroc = 0
            for epoch in range(num_epochs):
                # In each epoch, we do a full pass over the training data:
                train_err = 0
                train_batches = 0
                train_acc = 0
                start_time = time.time()

                train_targets = []
                train_preds = []

                if len(all_inputs) > 1:
                    print("\n Epoch {} of {}".format(epoch + 1, num_epochs))
                    print ' Training...'

                if len(all_inputs) == 1:
                    print("\n\n Epoch {} of {}".format(epoch + 1, num_epochs))
                    print ' Training...'

                for inputs, targets in iterate_minibatches(X_train, y_train, self.options["MINI_BATCH_SIZE"], True):
                    err, acc, protr = self.train_fn(inputs, targets)
                    train_err += err
                    train_acc += acc
                    train_batches += 1

                    train_targets.extend(targets)
                    train_preds.extend(protr)
                trtime = time.time() - start_time
                print(" Training took {:.3f}s".format(trtime))

                auroctr, _ = roc.get_auroc_data(train_targets, train_preds)
                train_all.append([train_err / train_batches, train_acc / train_batches * 100, auroctr])

                class0_prob = []
                class1_prob = []

                class0_seq = []
                class1_seq = []

                val_err = 0
                val_acc = 0
                val_batches = 0
                val_targets = []
                val_preds = []

                print ' Validating...'
                for inputs, targets in iterate_minibatches(X_val, y_val, self.options["MINI_BATCH_SIZE"], False):
                    err, acc, pro, eq = self.val_fn(inputs, targets)
                    val_err += err
                    val_acc += acc
                    val_batches += 1
                    val_targets.extend(targets)
                    val_preds.extend(pro)

                    for i in range(len(targets)):
                        if targets[i] == 0:
                            class0_prob.append(pro[i])
                            class0_seq.append(inputs[i])
                        if targets[i] == 1:
                            class1_prob.append(pro[i])
                            class1_seq.append(inputs[i])

                print(" Validating took {:.3f}s".format(time.time() - start_time - trtime))
                print(" Total epoch runtime was {:.3f}s".format(time.time() - start_time))
                sys.stdout.flush()

                auroctr, _ = roc.get_auroc_data(train_targets, train_preds)
                train_all.append([train_err / train_batches, train_acc / train_batches * 100, auroctr])
                auroc, _ = roc.get_auroc_data(val_targets, val_preds)
                test_all.append([time.time() - start_time, val_err / val_batches, val_acc / val_batches * 100, auroc])

                print("\n\tTraining loss:\t\t{:.6f}".format(train_err / train_batches))
                print("\tTraining accuracy:\t{:.4f} % ".format(train_acc / train_batches * 100))
                print("\tTraining AUROC:\t\t{:.4f} ".format(auroctr))
                print("\n\tValidation loss:\t{:.6f}".format(val_err / val_batches))
                print("\tValidation accuracy:\t{:.4f} % ".format(val_acc / val_batches * 100))
                print("\tValidation AUROC:\t{:.4f} ".format(auroc))
                sys.stdout.flush()

                b_cm1, b_cm2 = bernoulli_cm2_binary(class0_prob, class1_prob)
                cm_all.append([b_cm1, b_cm2])

                # extracting the best parameters based on highest bernoulli acc
                new_val = val_acc / val_batches * 100
                if new_val > best_val:
                    best_val = new_val
                    best_epoch_acc = epoch + 1
                    #best_network_data = self.options["file_name"] + "_slim_acc"
                    #self.save_slim_network(outfile=best_network_data, par=par)

                #print("\n Highest accuracy so far:\t{:.6f} from epoch {}".format(best_val, best_epoch_acc))

                loss = val_err / val_batches
                new_loss = loss
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_epoch_loss = epoch + 1
                    if self.options["par_selection"] == "loss":
                        # extracting the best parameters based on lowest loss
                        #new_loss = loss
                        #if new_loss < best_loss:
                        #    best_loss = new_loss
                        #    best_epoch_loss = epoch + 1
                        score = [['ep number:', epoch + 1], ['Acc:', best_val], ['Err:', val_err / val_batches], loss]
                        best_params = lasagne.layers.get_all_param_values(self.network['l_sumz'])
                        best_roc = _
                        #best_network_data = self.options["file_name"] + "_slim_auroc"
                        #self.save_slim_network(outfile=best_network_data, par=par)
                        #print(" Lowest loss so far:\t\t{:.4f} from epoch {}".format(best_loss, best_epoch_loss))
                        print " New parameters have been saved based on loss"
                        #if auroc >= self.options["auc_thr"] or auroctr >= self.options["auc_thr"]:
                        #    breaker += 1
                        #    if breaker >= 2:
                        #        print 'Breaking this run because 20 epochs produced either training or validation AUROC above {}'.format(self.options["auc_thr"])
                        #        break
                if self.options["par_selection"] == "loss":
                    if new_loss > best_loss:
                        early_stopping += 1
                    else:
                        early_stopping = 0
                        previous_loss = loss
                    if early_stopping >= self.options["early_stopping"]:
                        print "Breaking this training early because {} epochs in a row produced validation loss that were higher than the previous best loss score.".format(self.options["early_stopping"])
                        break

                print("\n Lowest loss so far:\t\t{:.4f} from epoch {}".format(best_loss, best_epoch_loss))
                print(" Highest accuracy so far:\t{:.6f} from epoch {}".format(best_val, best_epoch_acc))
                new_auroc = auroc
                if new_auroc > best_auroc:
                    best_auroc = new_auroc
                    best_epoch = epoch + 1
                    if self.options["par_selection"] == "auroc":
                        # extracting the best parameters based on highest AUROC
                        #new_auroc = auroc
                        #if new_auroc > best_auroc:
                        #    best_auroc = new_auroc
                        #    best_epoch = epoch + 1
                        score = [['ep number:', epoch + 1], ['Acc:', best_val], ['Err:', val_err / val_batches], auroc]
                        best_params = lasagne.layers.get_all_param_values(self.network['l_sumz'])
                        best_roc = _
                        #best_network_data = self.options["file_name"] + "_slim_auroc"
                        #self.save_slim_network(outfile=best_network_data, par=par)
                        print(" Highest AUROC so far:\t\t{:.4f} from epoch {}".format(best_auroc, best_epoch))
                        print " New parameters have been saved based on auroc"
                if auroc >= self.options["auc_thr"] or auroctr >= self.options["auc_thr"]:
                    breaker += 1
                    if breaker >= 2:
                        print 'Breaking this run because 20 epochs produced either training or validation AUROC above {}'.format(self.options["auc_thr"])
                        break
                if self.options["par_selection"] == "auroc":
                    if auroc < best_auroc:
                        early_stopping += 1
                    else:
                        early_stopping = 0
                        previous_auroc = auroc
                    if early_stopping >= self.options["early_stopping"]:
                        print "Breaking this training early because {} epochs in a row produced validation AUROC that didn't exceed the previous best AUROC score.".format(self.options["early_stopping"])
                        break
                print(" Highest AUROC so far:\t\t{:.4f} from epoch {}".format(best_auroc, best_epoch))

            print("\n Best validation loss:\t\t{:.4f} at epoch: {}".format(best_loss, best_epoch_loss))
            print(" Best validation accuracy:\t{:.6f} % at epoch: {}".format(best_val, best_epoch_acc))
            print(" Best validation AUROC:\t\t{:.4f} at epoch: {}".format(best_auroc, best_epoch))
            #print(" Best validation loss:\t\t{:.4f} at epoch: {}".format(best_loss, best_epoch_loss))


            sys.stdout.flush()
            lasagne.layers.set_all_param_values(self.network['l_sumz'], best_params)

            if self.options["runmode"] == "cv":
		print self.network['output_params']
                predict_fn, outpar = self.compile_prediction_function()
                save_network(self.network, self.options, self.options['cvfile'] + "_cv" + str(cf+1), [1,1,1,1])
                results = predict(self.network, self.options, predict_fn, X_test, self.network['output_params'])        
                cv_results.append(results)
                if self.options["par_selection"] == "auroc":
                    auc_cv.append(best_auroc)
                    roc_cv.append(best_roc)
                if self.options["par_selection"] == "loss":                    
                    auc_cv.append(best_loss)
                    roc_cv.append(best_roc)
                lasagne.layers.set_all_param_values(self.network['l_sumz'], init_params)
                cv_out = self.options['cvfile'] + "_cv" + str(cf+1) + "-predictions.txt"
                with open(cv_out, 'w') as outfile:
                    for i in range(len(y_test)):
                        line = str(int(y_test[i])) + " " + str(float(results["predictions"][i])) + "\n"
                        outfile.write(line)

#		f = open(cv_out[:-4]+'_id_seq_pred.txt', "w")
#    		for qm in range(len(te_sqs)):
#        	    start = next(j for j in range(len(te_sqs[qm])) if te_sqs[qm][j] != "n")
#        	    end = next(j for j in range(len(te_sqs[qm]),0,-1) if te_sqs[qm][j-1] != "n")
#        	    f.write(str(te_ids[qm])+'\t'+te_sqs[qm][start:end]+'\t'+str(float(results["predictions"][qm])))
#        	    f.write("\n")
#    		f.close()


                with open(self.options['cvfile'] + "_cv_cycle_data.pkl", "wb") as output:
                    pickle.dump([init_params, auc_cv], output, pickle.HIGHEST_PROTOCOL)
                print "\n Processing of CV set {} is complete".format(cf + 1), '\n'

        if self.options["runmode"] == "cv":
            return self.network, cv_results, auc_cv, roc_cv

        if self.options["runmode"] == "train":
            return self.network



def predict(net, options, predict_fn, inputs, outpar):
    return predict_without_network(predict_fn, options, (None, 1, inputs.shape[-1]), inputs, outpar)


def predict_without_network(predict_fn, options, output_shape, inputs, outpar, batchsize=None):
    """
        Returns a dict with different results, but always at least "predictions" (classifier) and "profiles" (binding profiles)
        Supports arbitrary batch_size (actually just predicts them all as one big batch)
    """
    if not batchsize:
        batchsize=options["MINI_BATCH_SIZE"]

    n_batch, n_features, onehot_length = output_shape
    inputs = inputs.reshape((-1, n_features, onehot_length))
    n_inputs = inputs.shape[0]
    #print ' Items in Test list: {}'.format((len(inputs),onehot_length))
    predictions = []
    argmaxs = []
    cnscores = []
    weights_par = []
    weights = []

    output_params = outpar.get_value()
    #output_params = output_params.reshape((-1, options["SEQ_SIZE"], 2 * options["N_LSTM"]))

    print " Predicting . . .",
    sys.stdout.flush()
    for batch in iterate_minibatches_2(inputs, batchsize):
        #print '.',
        #sys.stdout.flush()
        prediction, argmax, cnscore, weight = predict_fn(batch)

        #print '######', output_params, '#########'

        #weight = np.sum(np.reshape(weight, (-1, options["SEQ_SIZE"], len(options["VOCAB"]))), axis=-1)
        weight = np.reshape(weight, (-1, options["SEQ_SIZE"],1))# 2 * options["N_LSTM"] ))
        #weight = np.sum(np.reshape(weight, (-1, options["SEQ_SIZE"], 2 * options["N_LSTM"] )),axis=-1)
        weight_par = weight #np.sum(output_params * weight, axis=-1)
        #weight = np.sum(weight, axis=-1)
        #inp = np.sum(np.reshape(inputs, (-1, options["SEQ_SIZE"], len(options["VOCAB"]))), axis =-1)
        #weight = weight*inp
        #weight = weight/np.max(abs(weight))

        predictions.extend(prediction)
        argmaxs = np.append(argmaxs, argmax)
        cnscores = np.append(cnscores, cnscore)
        weights_par = np.append(weights_par, weight_par)
    	weights = np.append(weights, weight)
    print "done, completed",str(len(predictions)),"predictions."
    sys.stdout.flush()


    return {"predictions": predictions,
            "argmax": np.reshape(argmaxs, (-1, 1, options["ALL_F"])),
            "cnscore": np.reshape(cnscores, (-1, 1, options["ALL_F"])),
            "weights_par": np.reshape(weights_par, (-1, options["SEQ_SIZE"])),
	    "weights": np.reshape(weights, (-1, options["SEQ_SIZE"]))}


def save_prediction_function(net, outfile, freq):
    predict_fn, outpar = net.compile_prediction_function()
    with open(outfile, "wb") as output:
        pickle.dump([predict_fn, net.options, net.network['l_in'].output_shape, outpar, freq], output, pickle.HIGHEST_PROTOCOL)


def load_prediction_function(infile):
    with open(infile, "rb") as input:
        predict_fn, options, output_shape, outpar, freq = pickle.load(input)
    return predict_fn, options, output_shape, outpar, freq


def save_network(network, options, outfile, freq):
    params = lasagne.layers.get_all_param_values(network['l_out'])
    with open(outfile, "wb") as output:
        pickle.dump([params, options, freq], output, pickle.HIGHEST_PROTOCOL)


def load_network(infile):
    with open(infile, "rb") as input:
        params, options, freq = pickle.load(input)
        params32 = [p.astype(np.float32) for p in params]
        net = Network(**options)
        net.build_model()
        lasagne.layers.set_all_param_values(net.network['l_out'], params32)
    return net, freq


def load_params(infile):
    with open(infile, "rb") as input:
        params, options = pickle.load(input)
    #all_params = lasagne.layers.get_all_params(network.network['l_out'], trainable=True)
    #total_params = sum([p.get_value().size for p in all_params])
    #print " Total Model Parameters:", total_params
    for i in params:
	print i


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    """
    This function makes it possible to shuffle the train and validation data when
    iterating through it.
    It is a 100% copy of a function in the Lasagne tutorial.
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in xrange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def iterate_minibatches_2(inputs, batchsize):
    for start_idx in xrange(0, len(inputs), batchsize):
        excerpt = slice(start_idx, min([start_idx + batchsize, len(inputs)]))
        yield inputs[excerpt]
