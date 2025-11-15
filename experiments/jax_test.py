import jax
import jax.numpy as jnp
from enzyme_ad.jax import cpp_call

# Forward-mode C++ AD example

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")

@jax.jit
def something():
    ones = jnp.ones((2, 3), jnp.float32)
    shape = jax.core.ShapedArray(ones.shape, ones.dtype)
    a, b = cpp_call(
        ones,
        out_shapes=[shape, shape],
        source="""
        template<std::size_t N, std::size_t M>
        void myfn(enzyme::tensor<float, N, M>& out0,
                  enzyme::tensor<float, N, M>& out1,
                  const enzyme::tensor<float, N, M>& in0) {
          for (int j=0; j<N; j++) {
            for (int k=0; k<M; k++) {
                out0[j][k] = in0[j][k] + 42;
            }
          }
          for (int j=0; j<2; j++) {
            for (int k=0; k<3; k++) {
                out1[j][k] = in0[j][k] + 2 * 42;
            }
          }
        }
        """,
                fn="myfn",
        argv=argv,
    )
    return a, b

print(something())
