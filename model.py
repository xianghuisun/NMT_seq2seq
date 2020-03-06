import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self,encoder_gru_dim,encoder_vocab_size,embedding_dim,batch_size):
        super(Encoder,self).__init__()
        self.embedding_dim=embedding_dim
        self.encoder_gru_dim=encoder_gru_dim
        self.encoder_vocab_size=encoder_vocab_size
        self.batch_size=batch_size

        self.Embedding_layer=tf.keras.layers.Embedding(input_dim=self.encoder_vocab_size,output_dim=self.embedding_dim)
        self.GRU_layer=tf.keras.layers.GRU(units=self.encoder_gru_dim,return_sequences=True,return_state=True,
                recurrent_initializer="glorot_uniform")

    def call(self,input_tensor,hidden_state):
        #assert input_tensor.shape==(batch_size,en_max_length)
        embedding_out=self.Embedding_layer(input_tensor)
        #assert embedding_out.shape==(self.batch_size,en_max_length,self.embedding_dim)
        assert hidden_state.shape==(self.batch_size,self.encoder_gru_dim)
        gru_output,gru_state=self.GRU_layer(embedding_out,initial_state=hidden_state)
        return gru_output,gru_state

    def initialize_hidden_state(self):
        return tf.zeros(shape=[self.batch_size,self.encoder_gru_dim])

class BahdanauaAttention(tf.keras.Model):
    def __init__(self,fc_dim):
        super(BahdanauaAttention,self).__init__()
        self.fc_layer1=tf.keras.layers.Dense(units=fc_dim)
        self.fc_layer2=tf.keras.layers.Dense(units=fc_dim)
        self.fc_layer=tf.keras.layers.Dense(units=1)

    def call(self,decoder_hidden,encoder_outputs):
        #assert decoder_hidden.shape==(batch_size,decoder_gru_dim)
        #assert encoder_outputs.shape==(batch_size,en_max_length,encoder_gru_dim)
        decoder_hidden=tf.expand_dims(decoder_hidden,axis=1)
        #assert decoder_hidden.shape==(batch_size,1,decoder_gru_dim)
        layer1_out=self.fc_layer1(decoder_hidden)#(batch_size,1,fc_dim)
        layer2_out=self.fc_layer2(encoder_outputs)#(batch_size,en_max_length,fc_dim)
        layer_sum=tf.nn.tanh(layer1_out+layer2_out)#(batch_size,en_max_length,fc_dim)
        score=self.fc_layer(layer_sum)#(batch_size,en_max_length,1)
        #不看batch_size这个维度,score可以看成一个长度为en_max_length的vector,每个值就是权重,用它乘以encoder_outputs,
        #shape是(en_max_length,encoder_gru_dim)，也就是相当于w1*g1+w2*g2+...wn*gn
        #其中w1.....wn就是score,g1....gn就是en_max_length个长度为encoder_gru_dim的向量，表示每一个单词的特征，
        #乘积后的结果是一个长度为encoder_gru_dim的向量
        attention_weights=tf.nn.softmax(score,axis=1)#在en_max_length这个维度上求softmax
        context_vector=tf.reduce_sum(attention_weights*encoder_outputs,axis=1)#在en_max_length这个维度上求和
        #assert context_vector.shape==(batch_size,encoder_gru_dim)
        return context_vector,attention_weights


class Decoder(tf.keras.Model):
    def __init__(self,decoder_gru_dim,decoder_vocab_size,embedding_dim,batch_size,attention_fc_dim):
        super(Decoder,self).__init__()
        self.decoder_gru_dim=decoder_gru_dim
        self.decoder_vocab_size=decoder_vocab_size
        self.embedding_dim=embedding_dim
        self.batch_size=batch_size

        self.Embedding_layer=tf.keras.layers.Embedding(input_dim=self.decoder_vocab_size,output_dim=self.embedding_dim)
        self.GRU_layer=tf.keras.layers.GRU(units=self.decoder_gru_dim,return_sequences=True,return_state=True,
                recurrent_initializer="glorot_uniform")
        self.FC_layer=tf.keras.layers.Dense(units=self.decoder_vocab_size)
        self.attention_op=BahdanauaAttention(attention_fc_dim)

    def call(self,input_tensor,hidden_state,encoder_outputs):
        assert hidden_state.shape==(self.batch_size,self.decoder_gru_dim)
        #assert input_tensor.shape==(self.batch_size,1)
        #如果没有attention,那么encoder_outputs就应该是最后时间步的输出(batch_size,encoder_gru_dim)
        context_vector,attention_weights=self.attention_op(hidden_state,encoder_outputs)
        embedding_out=self.Embedding_layer(input_tensor)
        assert embedding_out.shape==(self.batch_size,1,self.embedding_dim)#1是因为decoder是单步输出,所以传入的length就是一个字
        context_vector=tf.expand_dims(context_vector,axis=1)#(batch_size,1,encoder_gru_dim)
        combine_vector=tf.concat(values=[context_vector,embedding_out],axis=-1)
        #assert combine_vector.shape==(self.batch_size,1,embedding_dim+encoder_gru_dim)
        gru_output,gru_state=self.GRU_layer(combine_vector)
        assert gru_output.shape==(self.batch_size,1,self.decoder_gru_dim) and gru_state.shape==(self.batch_size,self.decoder_gru_dim)
        fc_out=self.FC_layer(tf.reshape(tensor=gru_output,shape=(-1,self.decoder_gru_dim)))
        assert fc_out.shape==(self.batch_size,self.decoder_vocab_size)
        return fc_out,gru_state,attention_weights

def loss_function(real_data,predict_data):
    mask=tf.math.logical_not(tf.math.equal(real_data,0))#cn_word2id["<pad>"]==0
    mask=tf.cast(mask,dtype=tf.float32)#True will 1. and False will be 0
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')
    loss=loss_object(real_data,predict_data)#Don't miss the order
    loss=loss*mask
    return tf.reduce_mean(loss)




        
