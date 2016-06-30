# Graph
* without explicit specification, default graph, get by `tf.get_default_graph()`
* or create with `Graph.as_default()` context manager
```python
g = tf.Graph()
with g.as_default():
    # or `with tf.Graph.as_default() as g:`
    c = tf.constant(30)
    assert c.graph is g
```

* **Note**: not thread-safe. all operations should be created from single thread

# Session
* usage 
```python
ops = ...some graph...
sess = tf.Session()
result = sess.run(ops)
print result
sess.close()
```

* or use context manager `with tf.Session() as sess:`
* use `with tf.device('/gpu:0')` to specify assignment
* `tf.InteractiveSession()` works like IPython

# Tensor
* Tensor has ranks/shape/dtype ....

# Variable
* `update = tf.assign(old_value, new_value)`, then run `sess.run(update)`, the value is updated
    * for a variable, `sess.run(symbol)` will return the value
    * for an operation, `sess.run(op)` just update
* `sess.run(tf.initialize_all_variables())` to initialize all variables
* `w2 = tf.Variable(w1.initialized_value())` to initialize from another variable

# Saver
* save
```python
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
    sess.run(init_op)
    # Do some work with the model...
    # Save the variables to disk.
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
```
* restore
```python
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")
    # Do some work with the model
    ...
```
* `saver` can be specified with variables needed to be saved and restored
```python
saver = tf.train.Saver({'v1': v1, 'v2': v2})
saver = tf.train.Saver([v1, v2])
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})
```
* save parameters `saver.save(sess, 'model-name', global_step=step)`, then the model name
will be 'model-name-' + int(step)

# Optimizer
* naive usage
```python
optim = tf.train.Optimizer.GradientDescentOptimizer(learning_rate=0.01)
opt_op = optim.minimize(cost, var_list=<list of variables>)
sess.run(opt_op)
```
* processing gradients before applying them
```python
optim = tf.train.Optimizer.GradientDescentOptimizer(learning_rate=0.01)

grads_and_vars = optim.compute_gradients(loss, var_list=<list of variables>)

capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]

optim.apply_gradients(capped_grads_and_vars)
```
* learning rate decay
```python
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)
# Passing global_step to minimize() will increment it at each step.
learning_step = (tf.GradientDescentOptimizer(learning_rate)
                .minimize(...my loss..., global_step=global_step))
```

# Regularizer
* general regularization
```python
def regularize_cost(reg_func, name=None):
    G = tf.get_default_graph()
    params = G.get_collection(tf.GraphKeys.TRAINABLE_VARIABELS)

    reg_cost = []
    for p in params:
        if p.name.endswith('W'): # only regularize 'conv*/W' or 'fc*/W'
            reg_cost.append(reg_func(p))

    if not reg_cost:
        return 0
    return tf.add_n(reg_cost, name=name)
```
* for l2 regularization, use reg_func `tf.nn.l2_loss`

# Layers
* use `tf.contrib.layers` as the main codebase
* to regularize weights, these layers have `regularizer=` option

# Shared Variable
* `tf.get_variable(name, reuse=True)`
* useful when multi-gpu training and RNN network

# Summary

# QueneRunner & Coordinator
