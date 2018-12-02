1、使用grid的调lstm,cnn的参数有以下部分：（https://blog.csdn.net/wang1127248268/article/details/77200639）
下文所涉及的议题列表：

如何在scikit-learn模型中使用Keras。
如何在scikit-learn模型中使用网格搜索。
如何调优批尺寸和训练epochs。
如何调优优化算法。
如何调优学习率和动量因子。
如何确定网络权值初始值。
如何选择神经元激活函数。
如何调优Dropout正则化。
如何确定隐藏层中的神经元的数量。

```
调优后的lstm模型：
def create_model_lstm(neurons=60,dropout_rate=0.4, weight_constraint=2,activation='sigmoid',init_mode='lecun_uniform',learn_rate=0.001, momentum=0.4):
    # create model

   sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
   embedded_sequences = embedding_layer(sequence_input)
   embedded = Dropout(dropout_rate)(embedded_sequences)
   l_gru = Bidirectional(LSTM(neurons, return_sequences=True))(embedded_sequences)
   l_att = Attention_layer()(l_gru)
   dense_1 = Dense(neurons, init=init_mode,activation=activation,W_constraint=maxnorm(weight_constraint))(l_att)
   print("************************************************")
   print(dense_1.shape)
   dense_2 = Dense(2, init=init_mode,activation='sigmoid')(dense_1)
   print("************************************************")
   print(dense_2.shape)
   model = Model(sequence_input, dense_2)
   optimizer = SGD(lr=learn_rate, momentum=momentum)
   model.compile(loss=[focal_loss([6066, 1987])], optimizer=optimizer, metrics=[f1])
   return model
   
调优后的cnn模型：
def create_model_cnn(neurons=120,dropout_rate=0.0, weight_constraint=5,activation='linear',init_mode='lecun_uniform',learn_rate=0.2, momentum=0.4):
    # create model
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded = embedding_layer(sequence_input)
    embedded = Dropout(dropout_rate)(embedded)
    # convolutional layer
    convolution = Convolution1D(filters=nb_filter,
                                kernel_size=kernel_size,
                                padding='valid',
                                activation='relu',
                                strides=1
                                )(embedded)

    maxpooling = MaxPooling1D(pool_size=2)(convolution)
    maxpooling = Flatten()(maxpooling)

    # We add a vanilla hidden layer:
    dense = Dense(neurons,init=init_mode,activation=activation, W_constraint=maxnorm(weight_constraint))(maxpooling)  # best: 120
    dense = Dropout(dropout_rate)(dense)  # best: 0.25
    dense = Activation('relu')(dense)
    output = Dense(2, activation='sigmoid',init='lecun_uniform')(dense)
    model = Model(inputs=sequence_input, outputs=output)
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss=[focal_loss([6066, 1987])], optimizer=optimizer, metrics=[f1])
    return model
```
调参的代码：

```
'''
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(x_train, y_train_emb)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
'''
'''
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(x_train, y_train_emb)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_search.cv_results_['mean_test_score']
#params = grid_search.cv_results_['params']

for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
'''
'''
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

'''
'''
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

'''
'''
neurons = [60, 70, 80, 100, 120, 150]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

'''
```
2、将调优后的模型使用voting的soft的方式集成，效果不太好，感觉跟没有调参的效果差不多，预测的f1的值还是在0.67左右。

```
15	0.6612903226	submission.zip	11/29/2018 15:49:59	Finished		
16	0.5563636364	submission.zip	11/29/2018 16:25:31	Finished		
17	0.6666666667	submission.zip	11/30/2018 16:06:51	Finished		
18	0.6542372881	submission.zip	12/01/2018 10:49:12	Finished
```


```
# Use scikit-learn to grid search the batch size and epochs
clf1 = KerasClassifier(build_fn=create_model_lstm, verbose=2, epochs=10, batch_size=10)
clf2 = KerasClassifier(build_fn=create_model_cnn, verbose=2, epochs=cnn_epochs, batch_size=cnn_batch_size)
clf3 = KerasClassifier(build_fn=gru, verbose=2, epochs=10, batch_size=10)


eclf1 = voting_classifier.VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2),('clf3',clf3)], voting='soft')

#eclf1 = voting_classifier.VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2)], voting='soft')
eclf1.fit(x_train, y_train)
```
3、尝试用bert的代码进行测试。（使用run_classifier.py）
(1)、在flags中设置相关的运行参数
```
FLAGS = flags.FLAGS

# %%
## Required parameters
flags.DEFINE_string(
    "data_dir", "./data",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")


flags.DEFINE_string(
    "bert_config_file", "./bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "MRPC", "The name of the task to train.")

flags.DEFINE_string("vocab_file",  "./vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "./data",
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint",  "./bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 80,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size",8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 50,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 50,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

```
（2）使用MRPC的processor，设置label，参数和text参数

```
class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, labels):
        """See base class."""
        return ["0","1"]

    # new function get the result tsv
    def get_results(self, data_dir):
        """See base class."""
        return self._read_tsv(os.path.join(data_dir, "test_results.tsv"))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        labels = []
        labels_test = []
        for (i, line) in enumerate(lines,start=1):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)

            # tokenization is based on vocab file
            text_a = tokenization.convert_to_unicode(line[0])
            #print(text_a)
            label = tokenization.convert_to_unicode(line[1])
            #print(label)
            labels.append(label)
            
            if set_type == "test":
                label = "0"
            labels_test.append(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples, labels, labels_test
```
（3）打开文件时，修改部分参数

```
  def _read_tsv(cls, input_file, quotechar=None):
            """Reads a tab separated value file."""
            #with tf.gfile.Open(input_file,"rt",errors="ignore",encoding="utf-8") as f:
            reader = csv.reader(open(input_file, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
            #reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            print(len(lines))
            print(lines)
            return lines

```
4、运行bert的run_classifier.py脚本得到test_results.tsv，将第二列是分类的概率值，进行相应转换成labels。（使用label.py转换）
格式如下：
```
9.912115e-05,0.9999008
0.99997914,2.0845833e-05
0.9994112,0.0005887476

```

```
[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]

```
5、将label写入csv表，（使用predict_labels_to_csv.py）提交到codaLab，结果为

```
19	0.8468158348	submission.zip	12/02/2018 12:10:09	Finished
```
6、继续学习相关模型，进行相关的优化




