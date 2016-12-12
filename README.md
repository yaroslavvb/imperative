# imperative
imperative programming in TensorFlow

To install and test (needs Tensorflow 0.10):

    import urllib2
    response = urllib2.urlopen('https://raw.githubusercontent.com/yaroslavvb/imperative/georgia/release/imperative.py')
    with open("imperative.py", "w") as f:
      f.write(response.read())
    import imperative
    env = imperative.Env(tf)
    tfi = env.tf
    print(tfi.add(1, 2))
