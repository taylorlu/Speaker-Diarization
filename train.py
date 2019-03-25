# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A demo script showing how to use the uisrnn package on toy data."""

import numpy as np

import uisrnn


SAVED_MODEL_NAME = 'pretrained/saved_model.uisrnn_benchmark'

def similar(matrix):  # calc d-vectors similarity in pretty format output.
    ids = matrix.shape[0]
    for i in range(ids):
        for j in range(ids):
            dist = matrix[i,:]*matrix[j,:]
            dist = np.linalg.norm(matrix[i,:] - matrix[j,:])
            print('%.2f  ' % dist, end='')
            if((j+1)%3==0 and j!=0):
                print("| ", end='')
        if((i+1)%3==0 and i!=0):
            print('\n')
            print("*"*80, end='')
        print("\n")


def diarization_experiment(model_args, training_args, inference_args):
  """Experiment pipeline.

  Load data --> train model --> test model --> output result

  Args:
    model_args: model configurations
    training_args: training configurations
    inference_args: inference configurations
  """

  predicted_labels = []
  test_record = []

  train_data = np.load('./data/toy_training_data.npz')
  train_sequence = train_data['train_sequence']
  train_cluster_id = train_data['train_cluster_id']
  print("train_sequence = {}".format(train_sequence.shape))
  print("train_sequence = {}".format(train_sequence.dtype))
  print("train_cluster_id = {}".format(train_cluster_id.shape))


  train_data = np.load('./training_data.npz')
  train_sequence = train_data['train_sequence']
  train_cluster_id = train_data['train_cluster_id']
  train_sequence = train_sequence[:,0,:]
  train_sequence = train_sequence.astype(float)
  train_sequence += 0.00001
  train_cluster_id = train_cluster_id[:,0]
  print("train_sequence = {}".format(train_sequence.shape))
  print("train_sequence = {}".format(train_sequence.dtype))
  print("train_cluster_id = {}".format(train_cluster_id.shape))
  for i in range(10):
    print(np.linalg.norm(train_sequence[0,:]))

  model = uisrnn.UISRNN(model_args)

  # print(train_cluster_id[0], train_cluster_id[1], train_cluster_id[2])
  # print(train_cluster_id[120], train_cluster_id[121], train_cluster_id[122])

  # feats = []
  # feats += [train_sequence[0,:], train_sequence[1,:], train_sequence[2,:]]
  # feats += [train_sequence[1200,:], train_sequence[1201,:], train_sequence[1202,:]]
  # feats += [train_sequence[2400,:], train_sequence[2401,:], train_sequence[2402,:]]
  # feats += [train_sequence[3600,:], train_sequence[3601,:], train_sequence[3602,:]]
  # feats = np.array(feats)
  # similar(feats)
  # return

  # training
  # model.fit(train_sequence, train_cluster_id, training_args)
  # model.save(SAVED_MODEL_NAME)
  # we can also skip training by calling：
  model.load(SAVED_MODEL_NAME)

  
  test_sequence = train_sequence[1200::15, :]
  test_sequence = test_sequence[:40,:]
  test_sequence = test_sequence[::-1,:]
  print(test_sequence.shape)
  predicted_label = model.predict(test_sequence, inference_args)
  print(predicted_label)
  '''
  # testing
  for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
    predicted_label = model.predict(test_sequence, inference_args)
    predicted_labels.append(predicted_label)
    accuracy = uisrnn.compute_sequence_match_accuracy(
        test_cluster_id, predicted_label)
    test_record.append((accuracy, len(test_cluster_id)))
    print('Ground truth labels:')
    print(test_cluster_id)
    print('Predicted labels:')
    print(predicted_label)
    print('-' * 80)

  output_string = uisrnn.output_result(model_args, training_args, test_record)

  print('Finished diarization experiment')
  print(output_string)
  '''


def main():
  """The main function."""
  model_args, training_args, inference_args = uisrnn.parse_arguments()
  model_args.observation_dim = 512
  model_args.rnn_depth = 1
  model_args.rnn_hidden_size = 512
  training_args.enforce_cluster_id_uniqueness = False
  training_args.batch_size = 40
  training_args.learning_rate = 1e-3
  training_args.train_iteration = 50000
  # training_args.grad_max_norm = 5.0
  training_args.learning_rate_half_life = 8000
  diarization_experiment(model_args, training_args, inference_args)


if __name__ == '__main__':
  main()
