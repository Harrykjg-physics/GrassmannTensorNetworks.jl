using TupleTools

function TupleTools.deleteat(tup::NTuple{N, S}, ::Tuple{}) where {N, S}
    return tup
end

