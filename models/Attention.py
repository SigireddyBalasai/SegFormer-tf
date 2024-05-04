def call(self, x, H, W):
    get_shape = tf.shape(x)
    B = get_shape[0]
    C = get_shape[2]

    q = self.q(x)
    q = tf.reshape(q, (B, -1, self.num_heads, self.head_dim))
    q = tf.transpose(q, perm=[0, 2, 1, 3])

    if self.sr_ratio > 1:
        x = tf.reshape(x, (B, H, W, C))
        x = self.sr(x)
        x = tf.reshape(x, (B, -1, C))
        x = self.norm(x)

    k = self.k(x)
    k = tf.reshape(k, (B, -1, self.num_heads, self.head_dim))
    k = tf.transpose(k, perm=[0, 2, 1, 3])

    v = self.v(x)
    v = tf.reshape(v, (B, -1, self.num_heads, self.head_dim))
    v = tf.transpose(v, perm=[0, 2, 1, 3])

    attn = tf.matmul(q, k, transpose_b=True)  # Perform matrix multiplication with transpose
    scale = tf.cast(self.sqrt_of_units, dtype=attn.dtype)
    attn = tf.divide(attn, scale)

    attn = tf.nn.softmax(attn, axis=-1)
    attn = self.attn_drop(attn)
    x = tf.matmul(attn, v)
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    x = tf.reshape(x, (B, -1, self.units))
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
