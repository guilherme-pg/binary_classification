## Recognition of the Maracatu type



<p>The Maracatu is a combination of dance, musical rhythm and ritual of religious syncretism.</p>
<p>It is a cultural manifestation of Afro-Brazilian origin coming from the State of Pernambuco (Brazil), in which the conservation of this tradition predominates.</p>
<p>Maracatu is divided into two types (two beats): Maracatu Nação (Baque Virado) and Maracatu Rural (Baque Solto).</p>

<p>While the first is characterized by an association between percussion and a royal African procession practiced more frequently in the metropolitan regions of the State, the second is related to a rural tradition that links folklore with a game played by rural workers, involving, in addition to the essence of african elements, traditions from the Portuguese and indigenous peoples.</p>

<p>This project aims to create an algorithm that receives an image and responds if it is maracatu nation, rural maracatu or not maracatu.</p>





<p>CNNs (Convolutional Neural Network) use several filters, kernels or convolution matrices, resulting in the so-called convolutional layer to extract features from images.</p>

<p>Thus, the kernel moves around the image performing operations according to the number of stride steps (number of steps the kernel will pass through the pixels).</p>


<p>Keras Model</p>

<p>Sequential() is a class that allows building a layered sequential model.</p>

<p>Conv2D() is a layer that creates a convolution kernel, ie a convolution matrix for a two-dimensional input.</p>
<p>The 'filter' argument is used to designate the number of filters that will determine the number of kernels to convolutionize the input.</p>
<p>The 'kernel_size' is used to determine the dimensions of the kernel window.</p>
<p>The activation function being responsible for processing inputs, helping to provide an output. In which when using the "relu" (rectified linear unit') one tries to return positive values, returning zero if negative values are applied.</p>

<p>MaxPooling2D(), por sua vez, agrupa cada maior valor no mapa de recursos em um nova camada.</p>

<p>Flatten() is used for flattening through an extra layer. Thus, it converts a multidimensional matrix into a flat dimensional matrix, that is, one-dimensional.</p>

<p>Dense() </p>
<p>units </p>
<p>input_shape </p>
<p>activation </p>
<p></p>

<p>compile() </p>
<p>optimizer </p>
<p>loss </p>

<p>fit() </p>
<p>batch_size</p>
<p>echos </p>
<p>validation_split </p>

<p>in progress ....</p>
