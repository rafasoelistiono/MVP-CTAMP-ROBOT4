(define (stream ctamp)
  (:stream sample-grasp
    :inputs (?object)
    :domain (is-object ?object)
    :outputs (?grasp)
    :certified (is-grasp ?object ?grasp))

  (:stream sample-pick-pose
    :inputs (?object ?object-pose ?grasp)
    :domain (and (is-object ?object) (is-pose ?object-pose)
                 (is-grasp ?object ?grasp))
    :outputs (?pick-pose)
    :certified (and (grasp-pose ?object ?object-pose ?grasp ?pick-pose)
                    (is-pose ?pick-pose)))

  (:stream sample-place-pose
    :inputs (?object ?surface)
    :domain (and (is-object ?object) (is-surface ?surface))
    :outputs (?pose)
    :certified (and (placement ?object ?surface ?pose) (is-pose ?pose)))

  (:stream sample-stack-pose
    :inputs (?object ?support ?support-pose)
    :domain (and (is-object ?object) (is-object ?support)
                 (is-pose ?support-pose))
    :outputs (?pose)
    :certified (and (stack-placement ?object ?support ?support-pose ?pose)
                    (is-pose ?pose)))

  (:stream inverse-kinematics
    :inputs (?pose)
    :domain (is-pose ?pose)
    :outputs (?q)
    :certified (and (kinematic ?pose ?q) (is-config ?q)))

  (:stream plan-transit
    :inputs (?q1 ?q2)
    :domain (and (is-config ?q1) (is-config ?q2))
    :outputs (?trajectory)
    :certified (and (motion ?q1 ?q2 ?trajectory)
                    (collision-free ?q1 ?q2)
                    (is-trajectory ?trajectory)))

  (:stream test-motion
    :inputs (?q1 ?q2)
    :domain (and (is-config ?q1) (is-config ?q2))
    :outputs ()
    :certified (collision-free ?q1 ?q2))

  (:stream plan-pick
    :inputs (?object ?object-pose ?home)
    :domain (and (is-object ?object) (is-pose ?object-pose)
                 (home-config ?home))
    :outputs (?grasp ?pick-pose ?pick-q ?transit)
    :certified (and (pick-plan ?object ?object-pose ?home ?grasp ?pick-pose
                               ?pick-q ?transit)
                    (is-config ?pick-q) (is-trajectory ?transit)))

  (:stream plan-place
    :inputs (?object ?surface ?pick-q ?home)
    :domain (and (is-object ?object) (is-surface ?surface)
                 (is-config ?pick-q) (home-config ?home))
    :outputs (?pose ?place-q ?transfer ?return)
    :certified (and (place-plan ?object ?surface ?pick-q ?home ?pose ?place-q
                                ?transfer ?return)
                    (is-pose ?pose) (is-config ?place-q) (is-trajectory ?transfer)
                    (is-trajectory ?return)))

  (:stream plan-stack
    :inputs (?object ?support ?support-pose ?pick-q ?home)
    :domain (and (is-object ?object) (is-object ?support)
                 (is-pose ?support-pose) (is-config ?pick-q)
                 (home-config ?home))
    :outputs (?pose ?place-q ?transfer ?return)
    :certified (and (stack-plan ?object ?support ?support-pose ?pick-q ?home
                                ?pose ?place-q ?transfer ?return)
                    (is-pose ?pose) (is-config ?place-q) (is-trajectory ?transfer)
                    (is-trajectory ?return)))

)
