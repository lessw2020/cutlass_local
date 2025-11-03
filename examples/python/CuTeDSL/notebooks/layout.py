import cutlass
import cutlass.cute as cute

@cute.jit
def coalesce_example():
    "demonstrates coalesce operation flattening and combining modes"
    layout = cute.make_layout(
        (2, (1, 6)), stride=(1, (cutlass.Int32(6), 2))
    )
    result = cute.coalesce(layout)

    print(">>> Original:", layout)
    cute.printf(">?? Original: {}", layout)
    print(">>> Coalesced:", result)
    cute.printf(">?? Coalesced: {}", result)

if __name__ == "__main__":
    coalesce_example()