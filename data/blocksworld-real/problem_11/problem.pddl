(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        purple - block
        green - block
        red - block
        blue - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (on purple yellow)
        (clear purple)
        (ontable green)
        (on red green)
        (on blue red)
        (clear blue)
        (ontable orange)
        (clear orange)
        ; yellow -> purple
        ; green -> red -> blue
        ; orange
    )
    (:goal (and (on purple orange)
        (on blue purple)
        (on green blue)
        (on red yellow)))
)
    