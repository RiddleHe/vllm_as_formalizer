(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        blue - block
        purple - block
        green - block
        yellow - block
        red - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (on blue orange)
        (on purple blue)
        (clear purple)
        (ontable green)
        (on yellow green)
        (on red yellow)
        (clear red)
        ; orange -> blue -> purple
        ; green -> yellow -> red
    )
    (:goal (and (on blue purple)
        (on red yellow)
        (on green orange)))
)
    