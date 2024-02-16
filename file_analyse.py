from lib.analyser import Analyser

analyser = Analyser()
analyser.play("examples/samples/amol-p.wav")
analyser.start_reading("examples/samples/amol-p.wav")
analyser.loop()
