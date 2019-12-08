import edu.cmu.dynet.Expression.{
  binaryLogLoss,
  input,
  logistic,
  parameter,
  tanh
}
import edu.cmu.dynet.{
  internal,
  ComputationGraph,
  Dim,
  Expression,
  FloatVector,
  ParameterCollection,
  SimpleSGDTrainer
}
import edu.cmu.dynet.internal.DynetParams

object StaticXor {
  val HIDDEN_SIZE = 16
  val EXAMPLE_COUNT = 20000
  val EPOCH_SIZE = 100

  def main(args: Array[String]): Unit = {

    internal.dynet_swig.initialize(new DynetParams())

    val pc = new ParameterCollection()

    val w = pc.addParameters(Dim(Seq(HIDDEN_SIZE, 2)))
    val b = pc.addParameters(Dim(HIDDEN_SIZE))
    val v = pc.addParameters(Dim(1, HIDDEN_SIZE))
    val trainer = new SimpleSGDTrainer(pc)

    var total_loss = 0.0f
    var seen_instances = 0
    //val computationGraph = ComputationGraph
    //ComputationGraph.forBlock { implicit cg =>
    val x_values = new FloatVector(Seq(0f, 0f))
    val x =
      input(Dim(Seq(2)), x_values) //(cg)
    val pred_y =
      logistic(parameter(v) * tanh(parameter(w) * x + parameter(b)))

    val y_values = new FloatVector(Seq(0.0f))
    val y =
      input(Dim(Seq(1)), y_values)
    val loss_expr = binaryLogLoss(pred_y, y)

    println("Before training:")
    test(x_values, pred_y)

    generate_xor_examples(EXAMPLE_COUNT).map {
      case ((input1, input2), answer) => {
        x_values.update(0, input1.toFloat)
        x_values.update(1, input2.toFloat)
        y_values.update(0, answer.toFloat)
        seen_instances += 1
        total_loss += ComputationGraph.forward(loss_expr).toFloat()
        ComputationGraph.backward(loss_expr)
        trainer.update()
        if (seen_instances % EPOCH_SIZE == 0) {
          println(s"Average loss: ${total_loss / EPOCH_SIZE}")
          total_loss = 0
        }
      }
    }

    println("After training:")
    test(x_values, pred_y)
  }

  private def generate_xor_examples(num: Int): Seq[((Int, Int), Int)] = {
    for {
      _ <- 0 to num
      x1 <- Seq(0, 1)
      x2 <- Seq(0, 1)
      answer = if (x1 == x2) 1 else 0
    } yield {
      Tuple2(Tuple2(x1, x2), answer)
    }
  }

  private def test(x_values: FloatVector, pred_y: Expression): Unit = {
    x_values.update(0, 0.0f)
    x_values.update(1, 0.0f)
    println(s"0,0: ${ComputationGraph.forward(pred_y).toFloat()}")
    x_values.update(0, 0.0f)
    x_values.update(1, 1.0f)
    println(s"0,1: ${ComputationGraph.forward(pred_y).toFloat()}")
    x_values.update(0, 1.0f)
    x_values.update(1, 0.0f)
    println(s"1,0: ${ComputationGraph.forward(pred_y).toFloat()}")
    x_values.update(0, 1.0f)
    x_values.update(1, 1.0f)
    println(s"1,1: ${ComputationGraph.forward(pred_y).toFloat()}")
  }
}
