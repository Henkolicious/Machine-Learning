<h1>Convolutional Neural Network</h1>
<h2>Image classification with 6 classes</h2>
<p>Built with Keras, see <i>"Assignment 3.py"</i> for source code.</p>
<p><b>Proccess</b> >> Convolution > Max pooling > Flattening > Fully connect</p>
<br/>
<h1>Output:</h1>
<pre>
Classes are:  {'crayfish': 0, 'elephant': 1, 'flamingo': 2, 'hedgehog': 3, 'kangaroo': 4, 'leopards': 5}
</pre>
<pre>
Guess: crayfish. Actual = crayfish
Guess: hedgehog. Actual = crayfish
Guess: crayfish. Actual = crayfish
Guess: crayfish. Actual = crayfish
Guess: crayfish. Actual = crayfish
Guess: elephant. Actual = elephant
Guess: elephant. Actual = elephant
Guess: elephant. Actual = elephant
Guess: crayfish. Actual = elephant
Guess: leopards. Actual = elephant
Guess: flamingo. Actual = flamingo
Guess: hedgehog. Actual = flamingo
Guess: hedgehog. Actual = flamingo
Guess: flamingo. Actual = flamingo
Guess: flamingo. Actual = flamingo
Guess: leopards. Actual = hedgehog
Guess: hedgehog. Actual = hedgehog
Guess: hedgehog. Actual = hedgehog
Guess: hedgehog. Actual = hedgehog
Guess: hedgehog. Actual = hedgehog
Guess: kangaroo. Actual = kangaroo
Guess: kangaroo. Actual = kangaroo
Guess: kangaroo. Actual = kangaroo
Guess: leopards. Actual = kangaroo
Guess: crayfish. Actual = kangaroo
Guess: leopards. Actual = leopards
Guess: leopards. Actual = leopards
Guess: crayfish. Actual = leopards
Guess: leopards. Actual = leopards
Guess: crayfish. Actual = leopards
</pre>
<pre>
Image scaling:           64 x 64
Number of images:        65
Number of epochs:        200
Batch size / epoch:      5
Number of classes:       6
Total guess accuracy:    0.6666666666666666
Execution time:          388 seconds
</pre>
<br/>
<br/>
<h2>Other parameters</h2>
<pre>
Classes are:  {'crayfish': 0, 'elephant': 1, 'flamingo': 2, 'hedgehog': 3, 'kangaroo': 4, 'leopards': 5}
</pre>
<pre>
Guess: elephant. Actual = crayfish
Guess: crayfish. Actual = crayfish
Guess: crayfish. Actual = crayfish
Guess: crayfish. Actual = crayfish
Guess: crayfish. Actual = crayfish
Guess: elephant. Actual = elephant
Guess: elephant. Actual = elephant
Guess: elephant. Actual = elephant
Guess: crayfish. Actual = elephant
Guess: elephant. Actual = elephant
Guess: crayfish. Actual = flamingo
Guess: elephant. Actual = flamingo
Guess: flamingo. Actual = flamingo
Guess: flamingo. Actual = flamingo
Guess: flamingo. Actual = flamingo
Guess: elephant. Actual = hedgehog
Guess: elephant. Actual = hedgehog
Guess: elephant. Actual = hedgehog
Guess: flamingo. Actual = hedgehog
Guess: kangaroo. Actual = hedgehog
Guess: crayfish. Actual = kangaroo
Guess: crayfish. Actual = kangaroo
Guess: elephant. Actual = kangaroo
Guess: crayfish. Actual = kangaroo
Guess: elephant. Actual = kangaroo
Guess: elephant. Actual = leopards
Guess: elephant. Actual = leopards
Guess: elephant. Actual = leopards
Guess: elephant. Actual = leopards
Guess: crayfish. Actual = leopards
</pre>
<pre>
Image scaling:           128 x 128
Number of images:        65
Number of epochs:        100
Batch size / epoch:      1
Number of classes:       6
Total guess accuracy:    0.36666666666666664
Execution time:          258 seconds
</pre>
<br/>
<br/>
<h2>Other parameters</h2>
<pre>
Classes are:  {'crayfish': 0, 'elephant': 1, 'flamingo': 2, 'hedgehog': 3, 'kangaroo': 4, 'leopards': 5}
</pre>
<pre>
Guess: elephant. Actual = crayfish
Guess: kangaroo. Actual = crayfish
Guess: crayfish. Actual = crayfish
Guess: crayfish. Actual = crayfish
Guess: hedgehog. Actual = crayfish
Guess: elephant. Actual = elephant
Guess: elephant. Actual = elephant
Guess: elephant. Actual = elephant
Guess: kangaroo. Actual = elephant
Guess: hedgehog. Actual = elephant
Guess: flamingo. Actual = flamingo
Guess: hedgehog. Actual = flamingo
Guess: hedgehog. Actual = flamingo
Guess: flamingo. Actual = flamingo
Guess: flamingo. Actual = flamingo
Guess: hedgehog. Actual = hedgehog
Guess: hedgehog. Actual = hedgehog
Guess: hedgehog. Actual = hedgehog
Guess: hedgehog. Actual = hedgehog
Guess: hedgehog. Actual = hedgehog
Guess: kangaroo. Actual = kangaroo
Guess: kangaroo. Actual = kangaroo
Guess: kangaroo. Actual = kangaroo
Guess: hedgehog. Actual = kangaroo
Guess: hedgehog. Actual = kangaroo
Guess: kangaroo. Actual = leopards
Guess: leopards. Actual = leopards
Guess: hedgehog. Actual = leopards
Guess: leopards. Actual = leopards
Guess: leopards. Actual = leopards
</pre>
<pre>
Image scaling:           100 x 100
Number of images:        65
Number of epochs:        100
Batch size / epoch:      65
Number of classes:       6
Total guess accurasy:    0.6333333333333333
Execution time:          197 seconds
</pre>