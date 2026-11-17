# Speech script: Reading chest X-rays with a small CNN

About 9 minutes in total. Words in **bold** are the ones to lean on. Text in [brackets] is an action, not something to say.

---

## 1. Title (30 s)

Good morning, everyone. Our project is about teaching a computer to read chest X-rays.

We built a convolutional neural network, a CNN, completely from scratch. It looks at one chest X-ray and answers a single question: is this lung **normal**, or does it show **pneumonia**?

[Point to the red circle on the X-ray.] This cloudy patch is the kind of thing a doctor looks for. Over the next few minutes we'll show you how we taught a network to look for it too.

---

## 2. The problem (45 s)

Here is the task. On the left is a normal chest X-ray. The lungs are dark, because air lets X-rays pass through. On the right is a lung with pneumonia. See these white, cloudy areas? That is fluid and inflammation.

Why a CNN? Because pneumonia is a **local pattern**: a hazy patch in one part of the lung. A convolution filter is built to find exactly that kind of pattern straight from the pixels, and pooling makes it still work when the patient stands a little to one side. Nobody has to describe pneumonia to the computer by hand.

One more thing before we continue. We care about **recall** as much as accuracy. Sending a sick patient home is much worse than a false alarm, so we will always ask: how many sick patients did the model actually find?

---

## 3. The dataset (50 s)

Our data is 12,024 adult chest X-rays from the RSNA Pneumonia Detection Challenge. Radiologists labelled them. We took them from a public Kaggle collection and shrank each one to 128 by 128 pixels.

We used all 6,012 pneumonia images, and exactly 6,012 normal ones, picked at random with a fixed seed. So the two classes are perfectly balanced: **half sick, half healthy**.

Then we split the data three ways: 8,416 images to train on, 1,804 to tune on, and 1,804 kept completely hidden for the final test.

[Point to the red note.] One important choice: we only used images from the RSNA source. If sick and healthy X-rays came from different hospitals, the network could cheat by learning to recognise the hospital instead of the disease.

---

## 4. Preprocessing (45 s)

Every image gets the same treatment. We make it grayscale, cut out the centre square, resize it to 128 by 128, and scale the pixel values to between 0 and 1.

On training images only, we also **augment**: a small random rotation, shift and zoom. [Point to the left film.] This X-ray changes every second. The model sees a slightly different picture each epoch, so it can't just memorise them.

And we deliberately did **not** flip images. [Point to the crossed-out film.] In a chest X-ray the heart is on the left. A mirrored image is not a real patient.

Because the classes are 50/50, a lazy model that always answers "pneumonia" would score only 50%. So accuracy means something here.

---

## 5. Structure diagram (75 s) — the most important slide

This is the structure of our whole project. At the top you can see the five stages: dataset, preprocessing, the CNN model, training and testing, and finally saving and predicting. Each one is a file in our code.

Below is the network itself. [Point at the Input.] An X-ray goes in on the left, 128 by 128, one gray channel. Then it passes through **four convolution blocks**. [Point at a dark slab.] A convolution layer finds patterns. [Point at a light slab.] A pooling layer halves the image size, keeping only the strongest response.

Look at how the shapes change: the images get smaller, 128, 64, 32, 16, then 8 pixels, but we get more filters, from 16 up to 128. The network trades **where** for **what**.

After the fourth block, we flatten the result into one long list of 8,192 numbers. A dense layer of 128 neurons combines them, a second dense layer gives two scores, and softmax turns the scores into two probabilities that add up to 100%.

[Click one or two blocks.] You can click any block to see what it does.

---

## 6. The model (45 s)

In total the network has about 1.15 million weights. This table shows where they are.

And here is a surprise. [Point to the bars.] The four convolution blocks, which do all the "seeing", hold only **8.5%** of the weights. The first dense layer, the one that combines everything, holds **91.5%**.

That makes sense: it connects each of the 8,192 features to each of 128 neurons. But it also tells us the model is small. It is far smaller than the networks you hear about in the news, and it still runs on an ordinary laptop.

---

## 7. Training (50 s)

For training we used the Adam optimizer with a learning rate of 0.001, batches of 64 images, and 15 epochs. On a laptop CPU that took around 25 minutes.

After every epoch the model is checked on the validation set, and whenever it improves, we save the weights.

[Point to the charts.] On the left, the loss goes down. On the right, accuracy goes up. Training is the blue line and validation is orange. The best validation accuracy was **92.7%**, at epoch 11. It's circled, and that is the version we saved.

Notice that the orange line jumps around more than the blue one. That is normal for validation. Still, the two lines stay close together, so the model is learning real patterns and not just memorising.

---

## 8. Test results (60 s)

Now the honest part. We tested on 1,804 X-rays the model had never seen. Overall **accuracy is 90.1%**.

The confusion matrix shows what that means. [Point to the matrix.] Of 902 healthy people, 850 were correctly cleared and 52 got a false alarm. Of 902 sick patients, 775 were found and **127 were missed**.

So recall for pneumonia is **85.9%**, and for healthy patients 94.2%. When the model says "pneumonia", it is right 93.7% of the time.

We want to be clear about this: the 127 missed patients are the number we would most like to reduce. A real medical tool would need a much lower number. This is a class project, not a medical device.

---

## 9. Feature maps (50 s)

We wanted to understand what the network actually looks at, so we pulled out what each block "sees".

[Normal selected.] Each row is one convolution block, and each picture is one filter. In **block 1**, the filters react to edges and ribs. You can clearly see the shape of the rib cage. In **block 2**, they respond to bigger shapes and outlines. By **blocks 3 and 4**, the pictures are coarse and abstract. They don't look like a chest any more, but they hold the information the classifier needs.

[Click Pneumonia.] If we switch to a pneumonia X-ray, the early layers still see edges, but deeper layers respond differently, to the hazy regions. The network gets from pixels to a decision step by step.

---

## 10. Saving the model (35 s)

Once training is done, we save the model. Only the learned weights are stored, in one file of 4.6 megabytes. We don't save the code.

To use the model later, we create a fresh network and load the weights into it. That is exactly what our `predict.py` script does. [Point to the terminal.] Here it is on ten test images, with the class and the confidence for each.

And the web page you are looking at uses the **same** weights. Which brings us to the demo.

---

## 11. Live demo (60 s)

[Click a sample X-ray.] Let's classify an image. The network runs right here in the browser. The picture is scanned, the CNN computes, and we get a verdict with a probability for each class. [Read the result aloud.]

At the bottom you can see the first block's filters, computed just now. And we compare with PyTorch on the same file to show that the two give the same answer.

[Click a normal sample, then one more.] Let's try a normal one too.

[Optional: drop a new image in.] And this works for any chest X-ray image, not only our samples.

Remember: this is a class project and not a medical tool.

---

## 12. Takeaways (50 s)

To finish. What worked: a small CNN, trained from scratch on a CPU in about 25 minutes, reached **90.1% accuracy and 85.9% pneumonia recall** on completely unseen X-rays. We used balanced classes from a single source, so the score reflects the disease and not the hospital.

The limits matter too. At 128 by 128 pixels we lose fine detail a radiologist would use. All the images come from one dataset, so another hospital could give lower scores. And "lung opacity" is a sign of pneumonia, not a full diagnosis.

Next steps would be bigger images, more data sources, and heat maps that show **where** the model is looking.

Everything runs with the commands on the right. Thank you. We're happy to take your questions.

---

## Likely questions

- **Why not use a pre-trained network?** The task asks us to build our own CNN. Ours is small and fast, and it shows every part of the pipeline.
- **Why is test accuracy lower than the best validation score?** We picked the best epoch using validation, so that number is slightly optimistic. The test set was never used for choices, so 90.1% is the fairer figure.
- **How do you know it's not cheating?** All images come from one source (RSNA), classes are balanced, and the test set is separate from training.
- **How could you reduce the 127 misses?** Lower the decision threshold for pneumonia, train longer or with larger images, and add more data.
- **Why no flips?** The heart is on the left, so a mirrored X-ray is not realistic.
