from data_process import *
from model import *
import tensorflow as tf

file_path=""
lines=get_lines(file_path=file_path)
en_sentence_list,cn_sentence_list=split_en_cn(lines=lines)
en_list,cn_list=process_sentence_fn(en_sentence_list,cn_sentence_list)
en_word2id,cn_word2id=get_word2id(en_list,cn_list)

en_id2word={k:v for v,k in en_word2id.items()}
cn_id2word={k:v for v,k in cn_word2id.items()}

sorted_en_list,sorted_cn_list=sorted_sentence(en_list,cn_list)
en_id_list,cn_id_list=sentence_to_id(sorted_en_list,sorted_cn_list,en_word2id,cn_word2id)

epochs=50

def pad_sentence(en_id_list,cn_id_list):
    en_max_length=max(len(en_sentence) for en_sentence in en_id_list)
    cn_max_length=max(len(cn_sentence) for cn_sentence in cn_id_list)
    pad_en=tf.keras.preprocessing.sequence.pad_sequences(en_id_list,padding="post",value=en_word2id['<pad>'],maxlen=en_max_length)
    pad_cn=tf.keras.preprocessing.sequence.pad_sequences(cn_id_list,padding="post",value=cn_word2id['<pad>'],maxlen=cn_max_length)
    return pad_en,pad_cn,en_max_length,cn_max_length

pad_en,pad_cn,en_max_length,cn_max_length=pad_sentence(en_id_list,cn_id_list)
print(pad_en.shape,pad_cn.shape)
print(en_max_length,cn_max_length)


def get_train_valid_data(pad_en,pad_cn):
    from sklearn.model_selection import train_test_split
    train_data,valid_data,train_target,valid_target=train_test_split(pad_en,pad_cn,test_size=0.01)
    assert train_data.shape[0]==train_target.shape[0] and valid_data.shape[0]==valid_target.shape[0]
    return train_data,valid_data,train_target,valid_target

def make_dataset(pad_en,pad_cn,batch_size):
    train_data,valid_data,train_target,valid_target=get_train_valid_data(pad_en,pad_cn)
    train_dataset=tf.data.Dataset.from_tensor_slices((train_data,train_target))
    valid_dataset=tf.data.Dataset.from_tensor_slices((valid_data,valid_target))
    train_dataset=train_dataset.repeat(epochs).batch(batch_size=batch_size,drop_remainder=True)
    return train_dataset,valid_dataset

train_dataset,valid_dataset=make_dataset(pad_en,pad_cn,batch_size=100)

embedding_dim=128
encoder_gru_dim=64
decoder_gru_dim=encoder_gru_dim
encoder_vocab_size=len(en_word2id)
decoder_vocab_size=len(cn_word2id)

encoder=Encoder(encoder_gru_dim,encoder_vocab_size,embedding_dim,batch_size=100)
decoder=Decoder(decoder_gru_dim,decoder_vocab_size,embedding_dim,batch_size=100,attention_fc_dim=64)

optimizer=tf.keras.optimizers.Adam()
def train_step(input_data,target_data,encoder_hidden,batch_size):
    assert input_data.shape==(batch_size,en_max_length) and target_data.shape==(batch_size,cn_max_length) and encoder_hidden.shape==(batch_size,encoder_gru_dim)
    loss_=0.0
    with tf.GradientTape() as tape:
        encoder_outputs,encoder_state=encoder(input_data,encoder_hidden)#put into input_data and initial encoder_hidden
        assert encoder_outputs.shape==(batch_size,en_max_length,encoder_gru_dim)
        assert encoder_state.shape==(batch_size,encoder_gru_dim)
        decoder_hidden=encoder_hidden#get encoder_outputs and final encoder_state assign decoder hidden as decoder initial hidden
        for steps in range(target_data.shape[-1]-1):
            assert target_data[:,steps].shape==(batch_size,)
            decoder_input=tf.expand_dims(target_data[:,steps],axis=1)#from target_data get decoder input, which just a word
            assert decoder_input.shape==(batch_size,1)
            decoder_outputs,decoder_hidden,attention_weights=decoder(decoder_input,decoder_hidden,encoder_outputs)
            #put decoder input(which will through embedding layer), decoder_hidden(which is the final encoder hidden) and encoder_inputs(which
            # in shape (batch_size,en_max_length,encoder_gru_dim and represents all features of the whole sentence)) into decoder(decoder will call attention)
            loss_value=loss_function(target_data[:,steps+1],decoder_outputs)#get decoder outputs(which is a vector in length of cn_max_length(no softmax))
            loss_+=loss_value
        trainable_variables=encoder.trainable_variables+decoder.trainable_variables
        gradients=tape.gradient(target=loss_,sources=trainable_variables)
        optimizer.apply_gradients(zip(gradients,trainable_variables))
        return loss_/int(target_data.shape[-1]-1)

def train(batch_size=100):
    for epoch in range(50):
        encoder_hidden=encoder.initialize_hidden_state()
        epoch_loss=0.0
        for input_data,target_data in train_dataset.take(pad_en.shape[0]//batch_size):
            loss_val=train_step(input_data,target_data,encoder_hidden,batch_size=batch_size)
            epoch_loss+=loss_val
        if epoch%5==0:
            print("Epoch is %d and loss value is %f " % (epoch,epoch_loss/(pad_en.shape[0]//batch_size)))

def evaluate(input_sentence):
    attention_matrix=np.zeros((cn_max_length,en_max_length))
    input_sentence=input_sentence.lower().strip()
    input_sentence=re.sub(pattern=r"([?!.,])",repl=r" \1 ",string=input_sentence)
    input_sentence=re.sub(pattern=r"[' ']+",repl=r" ",string=input_sentence)
    print(input_sentence)
    input_list=input_sentence.strip().split()
    input_id_list=[en_word2id.get(word,en_word2id["<unk>"]) for word in input_list]
    pad_id_list=tf.keras.preprocessing.sequence.pad_sequences([input_id_list],padding="post",value=en_word2id['<pad>'],maxlen=en_max_length)
    input_data=tf.convert_to_tensor(pad_id_list)
    #把这句英文送进encoder中，return encoder_outputs,encoder_hidden
    #decoder_input就是<start>
    encoder_hidden=tf.zeros((1,encoder_gru_dim))
    encoder_outputs,encoder_hidden=encoder(input_data,encoder_hidden)
    assert encoder_outputs.shape==(1,en_max_length,encoder_gru_dim)
    #decoder_hidden用encoder_hidden初始化,再连同encoder_outputs,decoder_input送进decoder中
    decoder_input=tf.expand_dims([cn_word2id["<start>"]],axis=1)
    assert decoder_input.shape==(1,1)
    decoder_hidden=encoder_hidden
    predict_sequence=""
    for t in range(cn_max_length):
        decoder_outputs,decoder_hidden,attention_weights=decoder(decoder_input,decoder_hidden,encoder_outputs)
        assert decoder_outputs.shape==(1,decoder_vocab_size)
        attention_vector=tf.reshape(attention_weights,shape=(-1,))
        assert attention_vector.shape==(en_max_length,)
        attention_matrix[t]=attention_vector
        predict_id=tf.argmax(decoder_outputs,axis=-1).numpy().item()
        predict_word=cn_id2word[predict_id]
        if predict_word=="<end>":
            return predict_sequence,attention_matrix
        predict_sequence+=predict_word
        decoder_input=tf.expand_dims([predict_id],axis=0)
    return predict_sequence,attention_matrix


