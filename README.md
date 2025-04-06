# Ailie_Net
Ailie Net - A simplistic Python Neural Networks module, designed for beginners intending to explor AI from scratch.

This module is produced primarily on vanilla Python code, where external dependencies are kept to a bare minimum.
This vanilla-based approach is intended to increase compatibility with systems of varying Python version, operating systems, and architectures.

The only internal dependency of a third party library is that of NumPy, an extremely useful and flexible maths library at the heart of most Python projects.
Additional modules, such as those included with the test examples, include Matplotlib (data visualisation), Pandas (data managing), JSON (using JSON data files) and sys (for system commands).

# Build Instructions
    
    Create Python Virtual Environment
    ---------------------------------
        A virtual environment allows you to create a project, and install dependencies, isolated from
        other projects where differing package versions and variations have the potential to clash.
        
        First you need the tools to create a Python virtual environment. There are many popular options such as conda,
        venv (comes with versions of python) and virtualenv. To install virtualenv type the following into a prompt window:
            pip install virtualenv
        or:
            apt-get install virtualenv
        
        You now need a place to create your environment. You can do this by creating a new directory somewhere:
            cd Desktop
            mkdir ailie_build
            cd ailie_build
        
        With the tool installed we can now create an environment under a custom name. I am using the name "buildenv".
        Go to the projects intended directory and run:
            python -m venv buildenv
        
        Now the environment is created, we can activate it. This is done using the following command within the directory:
            source buildenv/bin/activate
        or on Windows:
            buildenv\Scripts\activate.bat
        
        The command prompt should now start with (buildenv) on each new prompt line. This shows that the new environment
        is activated and we are using it.
    
    
    Installing Building Tools
    -------------------------
        A series of packages are needed to create an installable python package.
        
        Enter the following into a python virtual environment:
            pip install setuptools wheel twine
        
        pip: A popular python package manager fpr installing new packages.
        setuptools: A package development tool for building and distributing packages.
        wheel: A format for distributing python modules in a ready to install approach.
        twine: A tool for publishing python modules to the Python Package Index (PyPI) registry.
        
        
    Building the Package
    --------------------
        Copy the package folder into the newly created Ailie_build directory.
        To build the package type the following:
            python setup.py sdist bdist_wheel
            
        Two new directories in the ailie_build folder should be created, build and dist.
        The dist folder contains the Python .whl that is our built module ready for isntall.
        
        
    Install the Created Package
    ---------------------------
        To install the package using pip, type the following command:
            pip install dist/<package_name>
        * Where <package_name> is the name of the package created for your systems setup.
        * The forward slash may have to be swapped to a backslash depending on OS
        
        On my laptop, this appears as: ailie_net-0.1-py3-none-any.whl
        
    
    Quick Test
    ----------
    
        If not already installed, numpy needs ot be installed in your environment
            pip install numpy
        
        In the command prompt activate a python terminal:
            python
        
        A new python terminal should now be created.
        You should now be able to import the Ailie_Net module using the following command:
            import Ailie_Net as ai
            
        If no error message is displayed, then the install is successful.
        
        The package also has a number of test scripts available. To run these the following dependencies
        may be required:
            pip install numpy
            pip install pandas
            pip install matplotlib
            pip install json


# Bibliography
Biological Neurology Background\
[1] Neuroscientifically Challenged, 2-Minute Neuroscience: Action Potential, (Jul. 26, 2014). Accessed: Jan. 29, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=W2hHt_PXe5o
[2] Neuroscientifically Challenged, 2-Minute Neuroscience: Divisions of the Nervous System, (Aug. 08, 2014). Accessed: Feb. 02, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=q3OITaAZLNc
[3] Neuroscientifically Challenged, 2-Minute Neuroscience: Membrane Potential, (Jul. 25, 2014). Accessed: Jan. 28, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=tIzF2tWy6KI
[4] Neuroscientifically Challenged, 2-Minute Neuroscience: Synaptic Transmission, (Jul. 22, 2014). Accessed: Jan. 28, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=WhowH0kb7n0
[5] Neuroscientifically Challenged, 2-Minute Neuroscience: The Neuron, (Jul. 22, 2014). Accessed: Jan. 28, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=6qS83wD29PY
[6] Neuroscientifically Challenged, 10-Minute Neuroscience: Neurons, (May 14, 2023). Accessed: Feb. 02, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=5p9ucgRDie8
[7] Harvard Online, How a synapse works, (Apr. 19, 2017). Accessed: Jan. 29, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=OvVl8rOEncE
[8] F. Amthor PhD, Neuroscience for Dummies, 3rd ed. in for Dummies. John Wiley & Sons, Inc, 2023.
[9] Daniel Kochli, PSY210 Ch2 Pt6: Synaptic Integration, (Aug. 28, 2020). Accessed: Feb. 02, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=Dw4d6zoWl9Q
[10] BioME, Temporal vs. Spatial Summation, (Jul. 30, 2020). Accessed: Feb. 02, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=KQOM_sXBtbw
[11] Bing Wen Brunton, Types of Synapses, (May 26, 2023). Accessed: Feb. 03, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=m4mNqY9iseE

Neural Networks Mathematics\
[1] 3Blue1Brown, Backpropagation calculus | DL4, (Nov. 03, 2017). Accessed: Mar. 12, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=tIeHLnjs5U8
[2] 3Blue1Brown, Backpropagation, step-by-step | DL3, (Nov. 03, 2017). Accessed: Mar. 12, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=Ilg3gGewQ5U
[3] 3Blue1Brown, But what is a neural network? | Deep learning chapter 1, (Oct. 05, 2017). Accessed: Mar. 12, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=aircAruvnKk
[4] Starfish Maths, Differentiation, (Oct. 25, 2016). Accessed: Mar. 10, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=BcOPKQAZcn0
[5] 3Blue1Brown, Gradient descent, how neural networks learn | DL2, (Oct. 16, 2017). Accessed: Mar. 12, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=IHZwWFHWa-w
[6] tecmath, How to do Calculus in Under 10 Minutes, (Mar. 06, 2024). Accessed: Mar. 10, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=aFczzFYjUec
[7] The Organic Chemistry Tutor, How To Multiply Matrices - Quick & Easy!, (Oct. 05, 2018). Accessed: Mar. 10, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=2spTnAiQg4M
[8] The Organic Chemistry Tutor, Intro to Matrices, (Feb. 16, 2018). Accessed: Mar. 10, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=yRwQ7A6jVLk
[9] Dr. Trefor Bazett, Multi-variable Optimization & the Second Derivative Test, (Nov. 24, 2019). Accessed: Mar. 11, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=_Ffcr98c7EE
[10] Bot Academy, Neural Networks Explained from Scratch using Python, (Jan. 30, 2021). Accessed: Mar. 11, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=9RN2Wr8xvro
[11] Khan Academy, Partial derivatives, introduction, (May 12, 2016). Accessed: Mar. 10, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=AXqhWeUEtQU
[12] Dr. Trefor Bazett, The Multi-Variable Chain Rule: Derivatives of Compositions, (Nov. 13, 2019). Accessed: Mar. 11, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=9yCtWfI_Vjg
[13] The Organic Chemistry Tutor, Understand Calculus in 35 Minutes, (Sep. 10, 2018). Accessed: Mar. 10, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=WsQQvHm4lSw

AI Coding in Python\
[1] The Coding Train, 10.4: Neural Networks: Multilayer Perceptron Part 1 - The Nature of Code, (Jun. 27, 2017). Accessed: Mar. 27, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=u5GAVdLQyIg
[2] First Principles of Computer Vision, Backpropagation Algorithm | Neural Networks, (Jun. 10, 2021). Accessed: Mar. 14, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=sIX_9n-1UbM
[3] DeepBean, Backpropagation: How Neural Networks Learn, (Feb. 27, 2023). Accessed: Mar. 15, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=fGDT63dZcvE
[4] Samson Zhang, Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math), (Nov. 24, 2020). Accessed: Mar. 12, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=w8yWXqWQYmU
[5] J. Portilla, ‘Complete Tensorflow 2 and Keras Deep Learning Bootcamp’, Udemy. Accessed: Feb. 17, 2025. [Online]. Available: https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/
[6] Python Simplified, Gradient Descent - Simply Explained! ML for beginners with Code Example!, (Aug. 09, 2021). Accessed: Mar. 13, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=jwStsp8JUPU
[7] NeuralNine, Gradient Descent From Scratch in Python - Visual Explanation, (Apr. 18, 2023). Accessed: Mar. 14, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=gsfbWn4Gy5Q
[8] Sebastian Lague, How to Create a Neural Network (and Train it to Identify Doodles), (Aug. 12, 2022). Accessed: Mar. 14, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=hfMk-kjRv4c
[9] D. Dato-On, ‘MNIST in CSV’, Kaggle.com. Accessed: Mar. 13, 2025. [Online]. Available: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
[10] The Independent Code, Neural Network from Scratch | Mathematics & Python Code, (Jan. 13, 2021). Accessed: Mar. 14, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=pauPCy_s0Ok
[11] Vizuara, Neural Network From Scratch: No Pytorch & Tensorflow; just pure math | 30 min theory + 30 min coding, (Dec. 27, 2024). Accessed: Mar. 14, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=A83BbHFoKb8
[12] M. A. Nielsen, ‘Neural Networks and Deep Learning’, 2015, Accessed: Mar. 14, 2025. [Online]. Available: http://neuralnetworksanddeeplearning.com
[13] Python Simplified, Perceptron Algorithm with Code Example - ML for beginners!, (Jun. 03, 2021). Accessed: Mar. 13, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=-KLnurhX-Pg
[14] The Independent Code, Softmax Layer from Scratch | Mathematics & Python Code, (Nov. 16, 2021). Accessed: Mar. 28, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=AbLvJVwySEo

Chatbot AI\
[1] freeCodeCamp.org, Create a Large Language Model from Scratch with Python – Tutorial, (Aug. 25, 2023). Accessed: Mar. 07, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=UU1WVnMk4E8
[2] Tech With Tim, Python Chat Bot Tutorial - Chatbot with Deep Learning (Part 1), (May 28, 2019). Accessed: Mar. 07, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=wypVcNIH6D4

Speech Recognition\
[1] NeuralNine, Speech Recognition in Python, (Mar. 28, 2021). Accessed: Mar. 09, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=9GJ6XeB-vMg

Python Programming\
[1] Indently, 5 Good Python Practices, (Sep. 09, 2024). Accessed: Mar. 17, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=plXuoYYFS-Y
[2] Travis Media, 5 Signs of an Inexperienced Self-Taught Developer (and how to fix), (Jan. 24, 2024). Accessed: Mar. 17, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=B_HR2R3xsnQ
[3] Tech With Tim, 5 Tips To Organize Python Code, (Mar. 05, 2022). Accessed: Mar. 17, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=e9yMYdnSlUA
[4] mCoding, 25 nooby Python habits you need to ditch, (Nov. 15, 2021). Accessed: Mar. 17, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=qUeud6DvOWI
[5] J. Portilla, ‘Complete Tensorflow 2 and Keras Deep Learning Bootcamp’, Udemy. Accessed: Feb. 17, 2025. [Online]. Available: https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/
[6] Python Simplified, Create Your Own Python PIP Package, (Jan. 03, 2025). Accessed: Mar. 17, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=9Ii34WheBOA
[7] LeMaster Tech, How to Connect and Control an Arduino From Python!, (Nov. 30, 2023). Accessed: Mar. 10, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=UeybhVFqoeg
[8] NeuralNine, Importing Your Own Python Modules Properly, (Jul. 06, 2022). Accessed: Mar. 17, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=GxCXiSkm6no
[9] Socratica, JSON in Python  ||  Python Tutorial  ||  Learn Python Programming, (Aug. 11, 2017). Accessed: Mar. 19, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=pTT7HMqDnJw
[10] ‘Loading a JSON File in Python – How to Read and Parse JSON’, freeCodeCamp.org. Accessed: Mar. 19, 2025. [Online]. Available: https://www.freecodecamp.org/news/loading-a-json-file-in-python-how-to-read-and-parse-json/
[11] ‘Read JSON file using Python’, GeeksforGeeks. Accessed: Mar. 19, 2025. [Online]. Available: https://www.geeksforgeeks.org/read-json-file-using-python/
[12] Tech With Tim, Write Python Code Properly!, (Aug. 24, 2021). Accessed: Mar. 17, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=D4_s3q038I0




