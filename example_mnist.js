'use strict'

const jimp = require("jimp")
const fs = require('fs')
const ndarray = require('ndarray')
const dtype = require('dtype')
const _ = require('lodash')
const menoh = require('menoh') // This menoh module

const MNIST_IN_NAME = "139900320569040"
const MNIST_OUT_NAME = "139898462888656"
const INPUT_IMAGE_LIST = [
    "./data/mnist/0.png",
    "./data/mnist/1.png",
    "./data/mnist/2.png",
    "./data/mnist/3.png",
    "./data/mnist/4.png",
    "./data/mnist/5.png",
    "./data/mnist/6.png",
    "./data/mnist/7.png",
    "./data/mnist/8.png",
    "./data/mnist/9.png"
]



// Load all image files.
const loadInputImages = async ()=> {
    return Promise.all(INPUT_IMAGE_LIST.map((filename) => jimp.read(filename)));
}

// Find the indexes of the k largest values.
const findIndicesOfTopK= (a, k) => {
    let outp = []
    for (let i = 0; i < a.size; i++) {
        outp.push(i) // add index to output array
        if (outp.length > k) {
            outp.sort((l, r) => { return a.get(r) - a.get(l); })
            outp.pop()
        }
    }
    return outp
}

console.log('Using menoh core version %s', menoh.getNativeVersion())

const categoryList = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

/*
*/

// Load ONNX file
let builder = null
const batchSize = INPUT_IMAGE_LIST.length

const init = async ()=> {
    builder = await menoh.create('./data/mnist/mnist.onnx')
    // Add input data
    builder.addInput(MNIST_IN_NAME, [
        batchSize,  // 10 images in the data
        1,          // number of channels
        28,         // height
        28          // width
    ])

    // Add output
    builder.addOutput(MNIST_OUT_NAME)
}

const main = async () => {
    try {
        // Build a new Model
        const model = builder.buildModel({
            backendName: 'mkldnn'
        })
    
        // Create a view for input buffer using ndarray.
        const iData = (()=> {
            const prof = model.getProfile(MNIST_IN_NAME);
            return ndarray(new (dtype(prof.dtype))(prof.buf.buffer), prof.dims);
        })()
    
        // Create a view for output buffer using ndarray.
        const oData = (() => {
            const prof = model.getProfile(MNIST_OUT_NAME);
            return ndarray(new (dtype(prof.dtype))(prof.buf.buffer), prof.dims);
        })()
    
    
        const imageList = await loadInputImages()
    
        imageList.forEach((image, batchIdx) => {
            // All the input images are already croped and resized to 28 x 28.
            // Now, copy the image data into to the input buffer in NCHW format.
            image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
                const val = image.bitmap.data[idx]
                iData.set(batchIdx, 0, y, x, val)
            })
        })
    
        // Run the model
        await model.run()
        for (let bi = 0; bi < batchSize; ++bi) {
            console.log('### Result for %s', INPUT_IMAGE_LIST[bi]);
    
            const topK = findIndicesOfTopK(oData.pick(bi, null), 10);
            topK.forEach((i) => {
                console.log('[%d] %f %s', i, oData.get(bi, i), categoryList[i]);
            })
        }
    } catch(e) {
        console.log('Error:', e)
    }
}
(async () => {
    await init()
    await main()
})
()





