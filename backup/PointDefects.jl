module PointDefects

   using JuLIP: deleteat!, JVecF, AbstractAtoms
   using JuLIP.ASE: rnn, extend!
   import JuLIPMaterials: cluster


   export PointDefect, Vacancy, SelfInterstitial

   abstract type PointDefect end
   immutable Vacancy <: PointDefect end
   immutable SelfInterstitial <: PointDefect end


   function cluster(species::AbstractString, R::Real, ::Type{Vacancy})
      at = cluster(species, R)
      deleteat!(at, 1)
      return at
   end

   interstitial_positions = Dict(
      "Si" => JVecF([-2.53312, 0.0, 0.0])
   )

   function cluster(species::AbstractString, R::Real, ::Type{SelfInterstitial})
      at = cluster(species, R)
      extend!(at, species, interstitial_positions[species])
      return at
   end

end
