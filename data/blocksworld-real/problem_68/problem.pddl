(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        yellow - block
        orange - block
        blue - block
        red - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (clear purple)
        (ontable yellow)
        (clear yellow)
        (ontable orange)
        (on blue orange)
        (on red blue)
        (clear red)
        (ontable green)
        (clear green)
        ; purple
        ; yellow
        ; orange -> blue -> red
        ; green
    )
    (:goal (and (on purple yellow)
        (on blue purple)
        (on green blue)
        (on red green)
        (on orange red)))
)
    