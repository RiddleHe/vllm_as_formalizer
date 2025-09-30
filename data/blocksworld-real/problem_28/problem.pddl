(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        blue - block
        green - block
        yellow - block
        red - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (on blue orange)
        (on green blue)
        (clear green)
        (ontable yellow)
        (clear yellow)
        (ontable red)
        (clear red)
        (ontable purple)
        (clear purple)
        ; orange -> blue -> green
        ; yellow
        ; red
        ; purple
    )
    (:goal (and (on red blue)
        (on purple red)
        (on green purple)
        (on yellow green)
        (on orange yellow)))
)
    